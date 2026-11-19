"""Launch the bound archive restore pilot once and preserve run evidence."""
import argparse
from importlib.metadata import version
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

from check_s3_actions import ROOT, digest, write_json
from ball_retention.archive_restore_plan import DESIGN, load, validate_case
from ball_retention.resources import admission, directory_bytes, snapshot

sys.path.insert(0, str(ROOT))
from diagnostics.resource_probe.runner import stop_owned_tree, storage_admission


OUT = ROOT/'runs/s3_archive_restore_pilot_20261114'


def run(out):
    start = time.monotonic()
    plan, cfg, _ = load(ROOT)
    limits = plan['budget']
    steps = limits['max_physics_substeps']
    validate_case(ROOT, cfg, steps)
    out.mkdir(exist_ok=False)
    run_dir = out/'worker'

    def guard():
        if (time.monotonic()-start >= limits['max_wall_s']
                or directory_bytes(out) > limits['max_new_output_mib']*1024**2):
            raise RuntimeError('archive restore round wall or new output cap')

    def verify_hashes(hashes):
        for path, expected in hashes.items():
            if digest(ROOT/path) != expected:
                raise ValueError(f'frozen archive restore input changed: {path}')

    try:
        hardware = snapshot(ROOT)
        admit = admission(hardware, cfg['budget'])
        storage = storage_admission(ROOT, out, cfg['budget'])
        write_json(out/'admission.json', {'hardware': hardware, 'hardware_admission': admit, 'storage': storage})
        if not admit['passed'] or not storage['passed']:
            raise RuntimeError('archive restore resource admission failed')
        config_path = out/'config.json'
        write_json(config_path, cfg)
        cfg_hash = digest(config_path)
        paths = sorted({ROOT/DESIGN, Path(__file__), ROOT/'scripts/check_s3_actions.py',
                        *(ROOT/'diagnostics/resource_probe').glob('*.py'),
                        *(ROOT/'src/ball_retention').glob('*.py')})
        hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
        write_json(out/'execution_plan.json', {'design': plan, 'case_config_sha256': cfg_hash,
                   'sources_sha256': hashes, 'packages': {package: version(package) for package in
                       ('newton', 'warp-lang', 'numpy', 'scipy', 'trimesh')}})
        verify_hashes(hashes)
        guard()
        command = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'),
                   '--mode', 's3-archive-restore', '--config', str(config_path), '--output', str(run_dir),
                   '--steps', str(steps)]
        print(f'Archive restore pilot launched. Fixed limit {steps} physical substeps.', flush=True)
        with (out/'launcher.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                while proc.poll() is None:
                    guard()
                    time.sleep(1)
            finally:
                if proc.poll() is None:
                    stop_owned_tree(proc)
                    proc.wait(timeout=15)
        result = json.loads((run_dir/'result.json').read_bytes())
        worker = result.get('worker') or {}
        if (proc.returncode or result['status'] != 'completed' or worker.get('status') != 'archive_restore_completed'):
            raise RuntimeError(worker.get('error') or result.get('stop_reason') or result['status'])
        if worker['physics_steps'] != steps or result['config_sha256'] != cfg_hash:
            raise ValueError('archive restore worker count or config differs from the plan')
        if len(worker['completed_branches']) != 6:
            raise ValueError('wrong completed nominal branch count')
        verify_hashes(hashes)
        if digest(config_path) != cfg_hash:
            raise ValueError('case config changed')
        load(ROOT)
        guard()
        write_json(out/'validation.json', {'status': 'completed', 'physics_steps': steps,
                   'wall_s': time.monotonic()-start, 'branches': worker['completed_branches'],
                   'live_equivalent': worker['live_equivalent'],
                   'live_position_resolution_passed': worker['live_position_resolution_passed'],
                   'historical_comparison': worker['historical_comparison'],
                   'control_accepted': False, 'test_episode_files_read': False,
                   'artifacts_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*'))
                                        if p.is_file()}})
    except BaseException as error:
        evidence = run_dir/'worker_result.json'
        try:
            last = json.loads(evidence.read_bytes())
        except (OSError, ValueError):
            last = {}
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error),
                   'wall_s': time.monotonic()-start, 'last_reported_physics_steps': last.get('physics_steps'),
                   'branches': last.get('completed_branches', []),
                   'control_accepted': False, 'test_episode_files_read': False,
                   'artifacts_sha256': {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*'))
                                        if p.is_file()}})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT/'runs')
    run(out)
