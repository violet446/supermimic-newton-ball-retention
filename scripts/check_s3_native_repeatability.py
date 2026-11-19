"""Run the frozen deterministic native replay probe with one monitored worker."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from check_s3_actions import ROOT, digest, write_json
from ball_retention.native_repeatability_plan import DESIGN, load
from ball_retention.resources import admission, directory_bytes, snapshot

sys.path.insert(0, str(ROOT))
from diagnostics.resource_probe.runner import stop_owned_tree, storage_admission


OUT = ROOT/'runs/s3_native_repeatability_20261114'


def run(out):
    started = time.monotonic()
    plan, config, _ = load(ROOT)
    dependencies = {name: version(name) for name in ('newton', 'warp-lang', 'numpy', 'scipy', 'trimesh')}
    if (dependencies['newton'], dependencies['warp-lang']) != ('1.6.0', '1.17.0'):
        raise ValueError('unexpected Newton or Warp build')
    budget = plan['budget']
    out.mkdir(exist_ok=False)
    worker_dir = out/'worker'

    def artifacts():
        return {path.relative_to(out).as_posix(): digest(path)
                for path in sorted(out.rglob('*')) if path.is_file()}

    def guard():
        if (time.monotonic()-started >= budget['max_wall_s']
                or directory_bytes(out) > budget['max_new_output_mib']*1024**2):
            raise RuntimeError('round wall or new-output cap exceeded')

    try:
        observed = snapshot(ROOT)
        hardware = admission(observed, config['budget'])
        storage = storage_admission(ROOT, out, config['budget'])
        write_json(out/'admission.json', {'resources': observed, 'hardware': hardware, 'storage': storage})
        if not hardware['passed'] or not storage['passed']:
            raise RuntimeError('native repeatability resource admission failed')
        config_path = out/'config.json'
        write_json(config_path, config)
        config_hash = digest(config_path)
        paths = {ROOT/DESIGN, Path(__file__).resolve(), ROOT/'scripts/check_s3_actions.py',
                 *sorted((ROOT/'src/ball_retention').glob('*.py')),
                 *sorted((ROOT/'diagnostics/resource_probe').glob('*.py'))}
        hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}
        write_json(out/'execution_plan.json', {'design': plan, 'packages': dependencies,
                   'source_sha256': hashes, 'case_config_sha256': config_hash})
        guard()
        steps = budget['max_physics_substeps']
        arguments = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'),
                     '--mode', 's3-native-repeatability', '--config', str(config_path),
                     '--output', str(worker_dir), '--steps', str(steps)]
        print(f'Starting native repeatability worker, capped at {steps} physical substeps.', flush=True)
        with (out/'launcher.log').open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                while process.poll() is None:
                    guard()
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    stop_owned_tree(process)
                    process.wait(timeout=15)
        result = json.loads((worker_dir/'result.json').read_bytes())
        worker = result.get('worker') or {}
        if process.returncode or result['status'] != 'completed' or worker.get('status') != 'native_repeatability_completed':
            raise RuntimeError(worker.get('error') or result.get('stop_reason') or result['status'])
        if worker['physics_steps'] != steps or result['config_sha256'] != config_hash:
            raise ValueError('worker steps or configuration differ from frozen plan')
        comparison = json.loads((worker_dir/'comparison.json').read_bytes())
        if comparison != worker['native_comparison'] or len(worker['completed_branches']) != budget['nominal_replays']:
            raise ValueError('worker comparison or branch evidence incomplete')
        if digest(config_path) != config_hash or any(digest(ROOT/name) != value for name, value in hashes.items()):
            raise ValueError('execution config or source changed during run')
        load(ROOT)
        guard()
        write_json(out/'validation.json', {'status': 'completed', 'physics_steps': steps, 'branches': worker['completed_branches'],
                   'passed': comparison['numerical_equivalent'] and comparison['position_resolution_passed'],
                   'comparison': comparison, 'control_accepted': False, 'test_episode_files_read': False,
                   'wall_s': time.monotonic()-started, 'artifacts_sha256': artifacts()})
    except BaseException as error:
        try:
            last = json.loads((worker_dir/'worker_result.json').read_bytes())
        except (OSError, ValueError):
            last = {}
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error),
                   'last_reported_physics_steps': last.get('physics_steps'), 'branches': last.get('completed_branches', []),
                   'wall_s': time.monotonic()-started, 'control_accepted': False, 'artifacts_sha256': artifacts()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT/'runs')
    run(out)
