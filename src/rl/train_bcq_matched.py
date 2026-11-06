"""Standalone matched-config BCQ run.

Same algorithm and analysis as train_bcq.py, but overrides the config to
match the CQL pipeline's oversampling setting (oversample_factor = 3). This
keeps a single source of truth for the model code while allowing the control
(oversample=1) and matched (oversample=3) runs to be launched as independent
commands. See train_bcq.py's module docstring for the algorithm caveats.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_RL = Path(__file__).resolve().parent
if str(_SRC_RL) not in sys.path:
    sys.path.insert(0, str(_SRC_RL))

import train_bcq as _tb  # noqa: E402


MATCHED_OVERSAMPLE = 3
MATCHED_MODEL_DIR = 'models_10seed/epsilon/bcq_matched'
MATCHED_OUT = 'reports/epsilon_bcq_matched_results.json'


def main():
    argv = sys.argv[1:]

    # Inject matched defaults only if the caller did not supply them.
    if '--oversample-factor' not in argv:
        argv += ['--oversample-factor', str(MATCHED_OVERSAMPLE)]
    if '--model-dir' not in argv:
        argv += ['--model-dir', MATCHED_MODEL_DIR]
    if '--out' not in argv:
        argv += ['--out', MATCHED_OUT]

    sys.argv = [sys.argv[0]] + argv
    _tb.main()


if __name__ == '__main__':
    main()