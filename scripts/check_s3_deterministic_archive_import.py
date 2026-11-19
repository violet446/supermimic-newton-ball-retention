"""Preflight and run the frozen deterministic nominal archive import once."""
import argparse
import ast
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from check_s3_actions import ROOT, digest, write_json
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.deterministic_archive_import_plan import DESIGN, MODE, NATIVE_RUN, load
from ball_retention.resources import admission, directory_bytes, snapshot

sys.path[:0] = [str(ROOT), str(ROOT/'diagnostics/resource_probe')]
from runner import stop_owned_tree, storage_admission


OUT = ROOT/'runs/s3_deterministic_archive_import_20261115'
COMPARISONS = ('native_reference_comparison', 'same_configuration_comparison',
               'historical_repeatability', 'historical_cross_configuration')


def backend_preflight(plan):
    """Read installed Python sources and GPU identity without loading the simulator."""
    lock = (ROOT/'requirements.lock').read_text(encoding='utf-8')
    packages = {name: version(name) for name in ('newton', 'warp-lang', 'numpy', 'scipy', 'trimesh')}
    locked = dict(re.findall(r'^([\w-]+)==([^\s]+)', lock, re.MULTILINE))
    runtime = plan['runtime']
    if (packages != locked or packages['newton'] != runtime['newton'] or packages['warp-lang'] != runtime['warp']):
        raise ValueError('installed dependency versions differ from the frozen environment')
    options = {}
    installed_backend_sources = {}
    for relative, class_name, member, constructor in (
        ('newton/_src/sim/collide.py', 'CollisionPipeline', 'deterministic', True),
        ('newton/_src/solvers/xpbd/solver_xpbd.py', 'SolverXPBD', 'deterministic', True),
        ('warp/config.py', 'DeterministicMode', runtime['xpbd_deterministic_mode'], False),
    ):
        package = 'newton' if relative.startswith('newton/') else 'warp-lang'
        source = distribution(package).locate_file(relative)
        tree = ast.parse(source.read_bytes(), filename=str(source))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        if constructor:
            init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
            supported = member in {arg.arg for arg in (*init.args.args, *init.args.kwonlyargs)}
        else:
            supported = any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == member
                            for t in node.targets) for node in cls.body)
        options[f'{class_name}.{member}'] = supported
        installed_backend_sources[relative] = digest(source)
    if not all(options.values()) or runtime['collision_pipeline_deterministic'] is not True:
        raise ValueError('installed backend does not expose both frozen deterministic settings')

    previous = json.loads((ROOT/NATIVE_RUN/'validation.json').read_bytes())
    log = _verified_bytes(ROOT, f'{NATIVE_RUN}/worker/worker.log',
                          previous['artifacts_sha256']['worker/worker.log']).decode('utf-8')
    match = re.search(r'^\s*"cuda:0"\s*:\s*"[^"]+"\s*\([^\n]*\bsm_(\d+)', log, re.MULTILINE)
    if match is None:
        raise ValueError('accepted native run did not record CUDA architecture')
    device = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,compute_cap',
                                      '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
    if len(device) != 1:
        raise ValueError('the nvidia-smi GPU cannot be unambiguously mapped to cuda:0')
    index, name, capability = (field.strip() for field in device[0].split(','))
    if index != '0' or capability.replace('.', '') != match[1]:
        raise ValueError('local GPU architecture differs from the accepted native run')
    return {'packages': packages, 'backend_options_available': options,
            'backend_sources_sha256': installed_backend_sources,
            'device': {'name': name, 'cuda_compute_capability': capability, 'native_sm': match[1]}}


def preflight(plan, cfg, rows, guard):
    import numpy as np
    from deterministic_archive_metrics import validate_recorded_replays
    from native_repeatability import _nominal_schedule
    from saved_native_snapshot import decode_snapshot

    backend = backend_preflight(plan)
    tick = plan['case']['tick']
    row = rows[tick]
    substeps = row['control_substeps']
    ref = plan['native_reference']
    _, nominal, command, commands = _nominal_schedule(ROOT, cfg, rows, tick, substeps, guard)
    recorded = [json.loads((ROOT/path).read_bytes()) for path in ref['replays']]
    old = sorted((branch for branch in row['branches'] if branch['axis'] is None and branch['offset'] == 0),
                 key=lambda branch: branch['repeat'])
    validate_recorded_replays(recorded, old, nominal, tick)
    for branch in recorded:
        for actual, expected in zip(branch['commands'], commands, strict=True):
            for field in ('q', 'qd'):
                np.testing.assert_array_equal(actual[field], expected[field])
    layout = json.loads((ROOT/ref['layout']).read_bytes())
    document = json.loads((ROOT/ref['snapshot']).read_bytes())
    decode_snapshot(document, layout, cfg, tick, substeps, command)
    if 'newton' in sys.modules or 'warp' in sys.modules:
        raise RuntimeError('static preflight imported physics')
    return {**backend, 'passed': True, 'seed': cfg['seed'], 'tick': tick, 'control_substeps': substeps,
            'nominal_commands_verified': len(commands), 'saved_array_layout': layout['state_arrays'],
            'physical_substeps': 0, 'worker_starts': 0, 'test_episode_files_read': False}


def run(out, *, preflight_only=False):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    worker_dir = out/'worker'

    def artifacts():
        return {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}

    def guard():
        if time.monotonic()-began >= limits['max_wall_s']:
            raise RuntimeError('round wall cap exceeded')
        if directory_bytes(out) > limits['max_new_output_mib']*1024**2:
            raise RuntimeError('round new-output cap exceeded')

    try:
        plan, cfg, rows = load(ROOT)
        limits = plan['budget']
        if preflight_only:
            write_json(out/'preflight.json', preflight(plan, cfg, rows, guard))
            guard()
            return

        # Release the CPU planner's NumPy/SciPy memory before checking worker admission.
        static_dir = out/'static'
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--preflight-only',
                        '--output', str(static_dir)], cwd=ROOT, check=True,
                       timeout=limits['max_wall_s']-(time.monotonic()-began),
                       creationflags=subprocess.CREATE_NO_WINDOW)
        (static_dir/'preflight.json').replace(out/'preflight.json')
        static_dir.rmdir()
        guard()

        resources = snapshot(ROOT)
        hardware = admission(resources, cfg['budget'])
        storage = storage_admission(ROOT, out, cfg['budget'])
        write_json(out/'admission.json', {'resources': resources, 'hardware': hardware, 'storage': storage})
        if not hardware['passed'] or not storage['passed']:
            raise RuntimeError('deterministic archive import resource admission failed')

        config = out/'config.json'
        write_json(config, cfg)
        config_hash = digest(config)
        paths = {ROOT/DESIGN, Path(__file__).resolve(), ROOT/'scripts/check_s3_actions.py',
                 *(ROOT/'src/ball_retention').glob('*.py'), *(ROOT/'diagnostics/resource_probe').glob('*.py')}
        hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(paths)}
        write_json(out/'execution_plan.json', {'design': plan, 'source_sha256': hashes,
                   'case_config_sha256': config_hash, 'preflight_sha256': digest(out/'preflight.json')})
        guard()
        steps = limits['max_physics_substeps']
        command = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'),
                   '--mode', MODE, '--config', str(config), '--output', str(worker_dir), '--steps', str(steps)]
        print(f'Launching one nominal replay worker, capped at {steps} physical substeps.', flush=True)
        with (out/'launcher.log').open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
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
        if (process.returncode or result['status'] != 'completed' or not result['worker_started']
                or worker.get('status') != 'deterministic_archive_import_completed'):
            raise RuntimeError(result.get('stop_reason') or worker.get('error') or result.get('error') or result['status'])
        if (worker['physics_steps'] != steps or len(worker['completed_branches']) != 6
                or result['config_sha256'] != config_hash or digest(config) != config_hash):
            raise ValueError('completed steps, branches or configuration differ from the frozen plan')
        if (any(digest(ROOT/name) != value for name, value in hashes.items())
                or any(hashes.get(name, plan['inputs_sha256'].get(name)) != value
                       for name, value in result['source_sha256'].items())):
            raise ValueError('execution sources or the prescribed path changed during the run')
        load(ROOT)
        comparisons = {key: worker[key] for key in COMPARISONS}
        if any(worker[key]['passed'] is not True for key in COMPARISONS[:3]):
            raise ValueError('completed worker did not pass all three frozen gates')
        guard()
        write_json(out/'validation.json', {'status': 'completed', 'passed': True,
                   'seed': cfg['seed'], 'tick': plan['case']['tick'], 'physics_steps': steps,
                   'nominal_prefix_substeps': 0, 'branches': worker['completed_branches'],
                   'comparisons': comparisons, 'control_accepted': False, 'test_episode_files_read': False,
                   'wall_s': time.monotonic()-began, 'artifacts_sha256': artifacts()})
    except BaseException as error:
        try:
            last = json.loads((worker_dir/'worker_result.json').read_bytes())
        except (OSError, ValueError):
            last = {}
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error),
                   'wall_s': time.monotonic()-began, 'last_reported_physics_steps': last.get('physics_steps'),
                   'branches': last.get('completed_branches', []), 'control_accepted': False,
                   'test_episode_files_read': False, 'artifacts_sha256': artifacts()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output, preflight_only=args.preflight_only)
