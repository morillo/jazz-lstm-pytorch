#!/usr/bin/env bash
# =============================================================
#  Jazz LSTM — PyTorch Setup for Mac
#
#  Usage (from the pytorch/ directory):
#    cd ~/Documents/giga/jazz_solo/pytorch
#    chmod +x setup_pt.sh
#    ./setup_pt.sh
#
#  What it does:
#    1. Verifies Python 3.11
#    2. Creates .venv-pt/ virtual environment (separate from TF)
#    3. Installs PyTorch 2.3.1
#    4. Installs all other dependencies
#    5. Registers a Jupyter kernel named "Jazz LSTM – PyTorch"
# =============================================================

set -euo pipefail

VENV=".venv"
KERNEL_NAME="jazz-lstm-pt"
KERNEL_DISPLAY="Jazz LSTM – PyTorch"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   Jazz LSTM · PyTorch Setup              ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Require Python 3.11 ────────────────────────────────
echo "▶  Checking Python 3.11..."

PYTHON=""
for cmd in python3.11 python3; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [ "$VER" = "3.11" ]; then
      PYTHON="$cmd"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "❌  Python 3.11 not found."
  echo "   brew install python@3.11"
  exit 1
fi
echo "   ✓  $PYTHON  →  $($PYTHON --version)"

# ── 2. Create virtual environment ────────────────────────
echo ""
echo "▶  Creating virtual environment: $VENV ..."
if [ -d "$VENV" ]; then
  echo "   (already exists — skipping)"
else
  $PYTHON -m venv "$VENV"
  echo "   ✓  Created"
fi

source "$VENV/bin/activate"
echo "   ✓  Activated"

# ── 3. Upgrade pip ───────────────────────────────────────
echo ""
echo "▶  Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✓  pip $(pip --version | awk '{print $2}')"

# ── 4. Install PyTorch ───────────────────────────────────
echo ""
echo "▶  Installing PyTorch 2.3.1..."
pip install torch==2.3.1 torchaudio==2.3.1 --quiet
echo "   ✓  torch 2.3.1 + torchaudio 2.3.1"

# ── 5. Install remaining dependencies ────────────────────
echo ""
echo "▶  Installing remaining dependencies..."
pip install \
  "numpy==1.26.4" \
  "matplotlib==3.9.0" \
  "music21==9.1.0" \
  "mido==1.3.2" \
  "pydub==0.25.1" \
  "ipython==8.25.0" \
  "ipykernel==6.29.5" \
  "jupyter==1.0.0" \
  "notebook==7.2.2" \
  --quiet
echo "   ✓  numpy, matplotlib, music21, mido, pydub, jupyter"

# ── 6. Register Jupyter kernel ───────────────────────────
echo ""
echo "▶  Registering Jupyter kernel..."
python -m ipykernel install --user \
  --name "$KERNEL_NAME" \
  --display-name "$KERNEL_DISPLAY"
echo "   ✓  Kernel: \"$KERNEL_DISPLAY\""

# ── 7. Smoke test ────────────────────────────────────────
echo ""
echo "▶  Running smoke test..."
python - <<'EOF'
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
EOF

# ── 8. Done ──────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║   ✅  PyTorch setup complete!            ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "  Open VS Code:   code ."
echo "  Open notebook:  Jazz_LSTM_PT.ipynb"
echo "  Select kernel:  $KERNEL_DISPLAY"
echo ""
echo "  To activate the venv manually:"
echo "    source $VENV/bin/activate"
echo ""
