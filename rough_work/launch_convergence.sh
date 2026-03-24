#!/bin/bash
# launch_convergence.sh — run convergence test on europa
# This wrapper avoids set -u issues with conda activate by explicitly unsetting
# the problematic flag around conda activation.

# Disable treating unbound vars as errors (conda activation has these)
set +u

source ~/anaconda3/etc/profile.d/conda.sh
conda activate inferences2

# Re-enable strict mode for our own code
set -eo pipefail

cd ~/data/PhD/Inferences || exit 1

# Run just N_d=1000 by patching ND_VALUES at runtime
python -u - <<'EOF'
import sys
from pathlib import Path

WORKSPACE_ROOT = Path('/home/users/scro4564/data/PhD/Inferences')
for repo_root in [WORKSPACE_ROOT / "pygeoinf", WORKSPACE_ROOT / "pygeoinf3D"]:
    s = str(repo_root)
    if repo_root.exists() and s not in sys.path:
        sys.path.insert(0, s)

# Override ND_VALUES to only run N_d=1000
import pygeoinf3D.rough_work.sola_convergence_test as _mod
EOF
