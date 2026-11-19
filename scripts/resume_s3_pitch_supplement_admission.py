"""Wait briefly for host memory without changing the frozen physics reservation."""
import argparse
import time
from pathlib import Path
from unittest.mock import patch

from collect_s3_pitch_training_supplement import ROOT, capture, readiness


START_RAM = 2*1024**3
WAIT_SECONDS = 45


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--preflight-sha256', required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if len(out.relative_to(ROOT/'runs').parts) != 1:
        parser.error('select a new run directly under runs')

    snapshot = capture.snapshot
    admission_pending = True

    def wait_for_available_ram(root):
        nonlocal admission_pending
        if not admission_pending:
            return snapshot(root)
        admission_pending = False
        deadline = time.monotonic()+WAIT_SECONDS
        sample = snapshot(root)
        while sample['available_ram_bytes'] < START_RAM and time.monotonic() < deadline:
            time.sleep(1)
            sample = snapshot(root)
        return sample

    original_worker = ROOT/'scripts/collect_s3_pitch_training_supplement.py'
    with patch.multiple(capture, load_ready=readiness, DESIGN_SHA=args.preflight_sha256,
                        __file__=str(original_worker), snapshot=wait_for_available_ram):
        capture.capture(out, args.preflight.resolve(), args.preflight_sha256)


if __name__ == '__main__':
    main()
