#!/usr/bin/env bash
# =============================================================
#  Jazz LSTM — PyTorch setup (uv)
#
#  Usage (from the pytorch/ directory):
#    ./setup_pt.sh
#
#  What it does:
#    1. Checks that uv is installed
#    2. `uv sync`: installs Python 3.11 (.python-version) and the
#       pinned dependencies from uv.lock into .venv/
#    3. Registers a Jupyter kernel named "Jazz LSTM – PyTorch"
#    4. Runs an import smoke test
# =============================================================

set -euo pipefail

KERNEL_NAME="jazz-lstm-pt"
KERNEL_DISPLAY="Jazz LSTM – PyTorch"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Jazz LSTM · PyTorch Setup (uv)         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Require uv ────────────────────────────────────────
if ! command -v uv &>/dev/null; then
  echo "❌  uv not found. Install it: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
echo "▶  uv $(uv --version | awk '{print $2}')"

# ── 2. Sync environment ──────────────────────────────────
echo ""
echo "▶  uv sync (Python 3.11 + pinned dependencies → .venv/)..."
uv sync
echo "   ✓  $(uv run python --version)"

# ── 3. Register Jupyter kernel ───────────────────────────
echo ""
echo "▶  Registering Jupyter kernel..."
uv run python -m ipykernel install --user \
  --name "$KERNEL_NAME" \
  --display-name "$KERNEL_DISPLAY"
echo "   ✓  Kernel: \"$KERNEL_DISPLAY\""

# ── 4. Smoke test ────────────────────────────────────────
echo ""
echo "▶  Running import smoke test..."
uv run python - <<'PY'
import torch
import numpy as np
import music21
import matplotlib
print(f"   torch      {torch.__version__}")
print(f"   numpy      {np.__version__}")
print(f"   music21    {music21.VERSION_STR}")
print(f"   matplotlib {matplotlib.__version__}")
device = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
t = torch.ones(3, 3, device=device)
print(f"   device     {t.device}")
print("   ✓  All imports OK")
PY

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ✅  Setup complete!                    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Open VS Code:   code ."
echo "  Open the notebook: Jazz_LSTM_PT.ipynb"
echo "  Select this kernel (top-right in VS Code): $KERNEL_DISPLAY"
echo "  (VS Code can also pick .venv/bin/python directly.)"
echo ""
