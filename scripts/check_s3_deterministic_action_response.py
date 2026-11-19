"""Run the frozen S3 action-response comparison with one monitored Newton worker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from check_s3_deterministic_archive_import import ROOT, digest, stop_owned_tree, storage_admission, write_json
from ball_retention.archive_restore_plan import _verified_bytes
from ball_retention.deterministic_response_plan import DESIGN, DESIGN_SHA256, MODE, PREFLIGHT, PREFLIGHT_SHA256, load
from ball_retention.resources import admission, directory_bytes, snapshot


OUT = ROOT/'runs/s3_deterministic_action_response_20261115'


def run(out):
    began = time.monotonic()
    out.mkdir(exist_ok=False)
    worker_dir = out/'worker'

    def file_hashes():
        return {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}

    def guard():
        if time.monotonic()-began >= limits['max_wall_s']:
            raise RuntimeError('response round wall cap exceeded')
        if directory_bytes(out) > limits['max_new_output_mib']*1024**2:
            raise RuntimeError('response round new-output cap exceeded')

    launched = False
    try:
        plan, cfg, _ = load(ROOT)
        limits = plan['budget']
        previous = json.loads(_verified_bytes(ROOT, PREFLIGHT, PREFLIGHT_SHA256))
        static_out = out/'static'
        subprocess.run([sys.executable, '-B', str(ROOT/'scripts/preflight_s3_deterministic_action_response.py'),
                        '--output', str(static_out)], cwd=ROOT, check=True,
                       timeout=min(plan['static_budget']['wall_s'], limits['max_wall_s']-(time.monotonic()-began)),
                       creationflags=subprocess.CREATE_NO_WINDOW)
        static_report = static_out/'preflight.json'
        fresh = json.loads(static_report.read_bytes())
        if (fresh['passed'] is not True or fresh['design_sha256'] != DESIGN_SHA256
                or fresh['artifacts_sha256'] != previous['artifacts_sha256']):
            raise ValueError('fresh controller commands or model predictions differ from frozen preflight')
        for name, expected in fresh['artifacts_sha256'].items():
            if digest(static_out/name) != expected:
                raise ValueError(f'changed static artifact: {name}')
        guard()

        resources = snapshot(ROOT)
        hardware = admission(resources, cfg['budget'])
        storage = storage_admission(ROOT, worker_dir, cfg['budget'])
        write_json(out/'admission.json', {'resources': resources, 'hardware': hardware, 'storage': storage})
        if not hardware['passed'] or not storage['passed']:
            raise RuntimeError('deterministic response resource admission failed')

        config = out/'config.json'
        write_json(config, cfg)
        config_hash = digest(config)
        hashes = {**fresh['source_sha256'], DESIGN: DESIGN_SHA256,
                  Path(__file__).resolve().relative_to(ROOT).as_posix(): digest(Path(__file__))}
        joint_path = cfg['prescribed']['joint_path']
        hashes[joint_path] = digest(ROOT/joint_path)

        def verify_sources():
            if any(digest(ROOT/name) != expected for name, expected in hashes.items()):
                raise ValueError('response execution sources changed')

        write_json(out/'execution_plan.json', {'design': plan, 'source_sha256': hashes,
                   'case_config_sha256': config_hash, 'frozen_preflight_sha256': PREFLIGHT_SHA256,
                   'fresh_preflight_sha256': digest(static_report)})
        verify_sources()
        guard()
        steps = limits['max_physics_substeps']
        command = [sys.executable, '-B', str(ROOT/'diagnostics/resource_probe/runner.py'), '--mode', MODE,
                   '--config', str(config), '--output', str(worker_dir), '--steps', str(steps)]
        print(f'Launching one deterministic response worker with a {steps}-substep cap.', flush=True)
        with (out/'launcher.log').open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            launched = True
            try:
                while process.poll() is None:
                    guard()
                    time.sleep(1)
            finally:
                if process.poll() is None:
                    stop_owned_tree(process)
                    process.wait(timeout=15)
        monitored = json.loads((worker_dir/'result.json').read_bytes())
        worker = monitored.get('worker') or {}
        if (process.returncode or monitored['status'] != 'completed' or not monitored['worker_started']
                or worker.get('status') != 'deterministic_action_response_completed'):
            raise RuntimeError(monitored.get('stop_reason') or worker.get('error') or monitored.get('error')
                               or monitored['status'])
        if (worker['physics_steps'] != steps or worker['nominal_prefix_substeps'] != 0
                or worker['completed_branches'] != plan['branch_order']
                or monitored['config_sha256'] != config_hash or digest(config) != config_hash
                or any(hashes.get(name) != value for name, value in monitored['source_sha256'].items())):
            raise ValueError('response worker scope or execution hashes differ from the frozen plan')
        comparison = json.loads((worker_dir/'response_comparison.json').read_bytes())
        if (worker['nominal_comparison']['passed'] is not True or worker['shared_prefix_checks']['passed'] is not True
                or comparison['primary'] != worker['primary'] or set(worker['primary']) != set(plan['directions'])):
            raise ValueError('response worker did not record the required gates or direction results')
        verify_sources()
        load(ROOT)
        guard()
        write_json(out/'validation.json', {'status': 'completed', 'case': plan['case'],
                   'execution_gates_passed': True, 'physics_steps': steps, 'nominal_prefix_substeps': 0,
                   'branches': worker['completed_branches'], 'nominal_comparison': worker['nominal_comparison'],
                   'shared_prefix_checks': worker['shared_prefix_checks'], 'primary': worker['primary'],
                   'control_accepted': False, 'test_episode_files_read': False,
                   'wall_s': time.monotonic()-began, 'artifacts_sha256': file_hashes()})
    except BaseException as error:
        try:
            last = json.loads((worker_dir/'worker_result.json').read_bytes())
        except (OSError, ValueError):
            last = {}
        write_json(out/'stopped.json', {'status': 'stopped', 'reason': str(error), 'runner_launched': launched,
                   'last_reported_physics_steps': last.get('physics_steps', None if launched else 0),
                   'branches': last.get('completed_branches', []), 'control_accepted': False,
                   'test_episode_files_read': False, 'wall_s': time.monotonic()-began, 'artifacts_sha256': file_hashes()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    output = parser.parse_args().output.resolve()
    if not output.relative_to(ROOT/'runs').parts:
        raise ValueError('choose a fresh run subdirectory')
    run(output)
