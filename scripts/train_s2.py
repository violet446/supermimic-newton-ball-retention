"""Fit the bounded CPU world model on the frozen S2 pilot split."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ball_retention.guard import process_memory
from ball_retention.resources import directory_bytes, snapshot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def fit(cfg, out, started):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    from ball_retention.model_data import prepare_dataset
    from ball_retention.model_eval import evaluate_termination, evaluate_windows
    from ball_retention.world_model import Adam, ResidualMLP

    budget = cfg['budget']
    gib = 1024**3
    limits = {'rss_bytes':budget['max_process_rss_gib']*gib,
              'private_commit_bytes':budget['max_process_commit_gib']*gib}

    def check_resources(update):
        resource = snapshot(ROOT)
        memory = process_memory(os.getpid())
        elapsed = time.monotonic()-started
        output_bytes = directory_bytes(out)
        row = {'update':update, 'elapsed_s':elapsed, 'output_bytes':output_bytes, **resource, **memory}
        with (out/'resources.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        free_ram = budget['min_start_ram_gib'] if update < 0 else budget['min_running_ram_gib']
        if (elapsed >= budget['wall_s'] or resource['available_ram_bytes'] < free_ram*gib
                or resource['disk_free_bytes'] < budget['min_disk_free_gib']*gib
                or any(memory[k] > limit for k, limit in limits.items())
                or output_bytes > budget['max_new_output_mib']*1024**2):
            raise RuntimeError('CPU training resource limit reached')

    check_resources(-1)
    data = prepare_dataset(ROOT/cfg['dataset'], horizons=tuple(cfg['horizons']))
    model = ResidualMLP(len(data.known_names), data.state_mean, data.state_scale, data.dt_s,
                        cfg['seed'], hidden=cfg['hidden'], residual_gain=cfg['residual_gain'])
    optimizer = Adam(model.params, learning_rate=cfg['learning_rate'], max_grad_norm=cfg['gradient_clip_norm'])
    rng = np.random.default_rng(cfg['seed'])
    weights = np.asarray(cfg['component_weights'])

    def normalized(windows):
        return ((windows.initial-data.state_mean)/data.state_scale,
                (windows.known-data.known_mean)/data.known_scale,
                (windows.target-data.state_mean)/data.state_scale)

    train = {h:normalized(w) for h,w in data.train.items()}
    validation = {h:normalized(w) for h,w in data.validation.items()}

    def selection_score():
        loss = []
        for initial, known, target in validation.values():
            error = model.rollout(initial, known)[:, -1]-target[:, -1]
            loss.append(float(np.average(np.mean(error**2, axis=0), weights=weights)))
        return float(np.mean(loss))

    best_score = initial_score = selection_score()
    best_update = 0
    best_state = model.state_dict()
    stale = 0
    last_update = 0
    history = out/'training.jsonl'
    check_resources(0)
    with history.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps({'update':0,'validation_score':initial_score})+'\n')
        for update in range(1, cfg['max_updates']+1):
            if time.monotonic()-started >= budget['wall_s']:
                raise RuntimeError('CPU training wall budget reached')
            horizon = cfg['horizons'][(update-1)%len(cfg['horizons'])]
            initial, known, target = train[horizon]
            idx = rng.integers(len(initial), size=cfg['batch_size'])
            loss, grads = model.loss_and_grad(initial[idx], known[idx], target[idx], state_weights=weights)
            grad_norm = optimizer.step(grads)
            if not np.isfinite(loss) or not np.isfinite(grad_norm):
                raise FloatingPointError('non-finite training loss or gradient')
            record = {'update':update, 'horizon':horizon, 'train_loss':float(loss), 'grad_norm':float(grad_norm)}
            last_update = update
            if update%cfg['validate_every']==0 or update==cfg['max_updates']:
                check_resources(update)
                score = selection_score()
                if not np.isfinite(score):
                    raise FloatingPointError('non-finite validation loss')
                record['validation_score'] = score
                if score < best_score-1e-8:
                    best_score, best_update = score, update
                    best_state = model.state_dict()
                    stale = 0
                else:
                    stale += 1
                print(f'update {update}: validation {score:.6g}, best {best_score:.6g}', flush=True)
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            if stale >= cfg['patience_validations']:
                break

    model = ResidualMLP.from_state_dict(best_state)
    checkpoint = out/'world_model.npz'
    np.savez_compressed(checkpoint, **best_state, known_mean=data.known_mean, known_scale=data.known_scale,
                        known_names=np.array(data.known_names), best_update=np.array(best_update))
    with np.load(checkpoint, allow_pickle=False) as saved:
        restored = ResidualMLP.from_state_dict(saved)
    initial, known, _ = validation[8]
    np.testing.assert_array_equal(model.rollout(initial[:2], known[:2]), restored.rollout(initial[:2], known[:2]))
    evaluation = {}
    for split, windows_by_horizon in [('train',data.train), ('validation',data.validation)]:
        evaluation[split] = {}
        for h, windows in windows_by_horizon.items():
            initial, known, _ = normalized(windows)
            prediction = restored.rollout(initial, known)*data.state_scale+data.state_mean
            evaluation[split][str(h)] = evaluate_windows(prediction, windows.target, windows.initial, data.dt_s,
                                                         windows.episode_ids, leads=(h,))

    # A timer-only probe checks that shortened terminal rows retain their event
    # label. It uses current elapsed outside time, independent of actual next dt.
    def ending_inputs(table):
        timer = np.asarray(table.departure_outside_s)
        labels = np.asarray(table.terminated, dtype=float)
        return np.column_stack((np.ones_like(timer), timer/.05)), labels

    x, labels = ending_inputs(data.train_endings)
    prior = float(labels.mean())
    if not 0 < prior < 1:
        raise ValueError('the ending probe requires both labels in training')

    def logistic_objective(coef):
        logit = x@coef
        penalty = cfg['ending_l2']*coef[1]**2/2
        loss = np.mean(np.logaddexp(0, logit)-labels*logit)+penalty
        grad = x.T@(expit(logit)-labels)/len(labels)+[0, cfg['ending_l2']*coef[1]]
        return loss, grad

    probe = minimize(logistic_objective, [np.log(prior/(1-prior)), 0.], jac=True, method='L-BFGS-B',
                     options={'maxiter':100, 'gtol':1e-9})
    if not probe.success or not np.isfinite(probe.x).all():
        raise RuntimeError('ending probe did not converge')
    ending = {'input':'current outside timer / 0.05s', 'coefficients':probe.x.tolist(),
              'training_prior':prior, 'scope':'one training and one validation event'}
    for split, table in [('train',data.train_endings), ('validation',data.validation_endings)]:
        x_split, label = ending_inputs(table)
        ending[split] = evaluate_termination(expit(x_split@probe.x), label, prior, episode_ids=table.episode_ids)

    result = {'status':'completed', 'updates':last_update, 'best_update':best_update,
              'validation_score_before_training':initial_score, 'best_validation_score':best_score,
              'checkpoint':checkpoint.name, 'checkpoint_sha256':digest(checkpoint), 'checkpoint_reload_exact':True,
              'known_features':list(data.known_names), 'horizon_selection':'equal mean of weighted final-state normalized MSE at 1/4/8 steps',
              'data':data.report, 'predictions':evaluation, 'ending_probe':ending,
              'wall_s':time.monotonic()-started}
    check_resources(last_update)
    write_json(out/'summary.json', result)
    print(f'completed {last_update} updates, checkpoint at {best_update}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/s2_world_model_20261112')
    args = parser.parse_args()
    cfg_path = ROOT/'configs/s2_world_model.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    out = (ROOT/args.output).resolve()
    out.relative_to(ROOT/'runs')
    manifest = ROOT/cfg['dataset']/'manifest.json'
    if digest(manifest) != cfg['dataset_manifest_sha256']:
        raise ValueError('dataset manifest changed')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(cfg['budget']['threads'])
    out.mkdir(parents=True, exist_ok=False)
    sources = [cfg_path, Path(__file__), *(ROOT/'src/ball_retention').glob('*.py')]
    write_json(out/'plan.json', {'config':cfg, 'fit_split':'train', 'selection_split':'validation',
        'test_used_for_model_evaluation':False, 'device':'cpu',
        'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','scipy')},
        'sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in sources}})
    started = time.monotonic()
    try:
        fit(cfg, out, started)
    except Exception as exc:
        write_json(out/'failure.json', {'status':'failed', 'error':str(exc), 'wall_s':time.monotonic()-started})
        raise


if __name__ == '__main__':
    main()
