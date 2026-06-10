# Jazz LSTM — PyTorch

A LSTM-based jazz music generator built with PyTorch. Trained on Pat Metheny's "Meet the Metheny" and capable of generating original jazz solos in the same style.

This is the **PyTorch version**. For the TensorFlow version, see the companion repo [`jazz-lstm-tensorflow`](https://github.com/morillo/jazz-lstm-tensorflow).

---

## What it does

The model learns the harmonic and rhythmic patterns of jazz improvisation as a sequence generation problem — the same architecture behind modern voice AI. An LSTM reads a sequence of musical "tokens" (chord + note abstractions), learns their probability distribution, then generates new sequences autoregressively at inference time.

GPU acceleration is used automatically when available (Apple Silicon, CUDA, or CPU fallback).

---

## Requirements

- macOS (Apple Silicon or Intel), Linux, or Windows
- Python 3.11
- [Homebrew](https://brew.sh) (macOS)

---

## Setup

```bash
cd jazz-lstm-pytorch
chmod +x setup_pt.sh
./setup_pt.sh
```

The script:
1. Creates `.venv/` with Python 3.11
2. Installs PyTorch 2.3.1 + torchaudio
3. Installs all pinned dependencies (numpy, matplotlib, music21, mido, pydub, jupyter)
4. Registers a Jupyter kernel named **"Jazz LSTM – PyTorch"**

---

## Run

Open in VS Code:

```bash
code .
```

1. Open `Jazz_LSTM_PT.ipynb`
2. Select kernel: **Jazz LSTM – PyTorch**
3. Run All Cells

Generated MIDI is written to `output/my_music.midi`. Audio rendering requires [MuseScore](https://musescore.org/) or [FluidSynth](https://www.fluidsynth.org/).

---

## Project structure

```
jazz-lstm-pytorch/
├── Jazz_LSTM_PT.ipynb     # Main notebook
├── setup_pt.sh            # One-command environment setup
├── requirements.txt       # Pinned dependencies
├── data_utils.py          # Dataset loading, preprocessing, predict_and_sample
├── grammar.py             # Music grammar rules (chord/note abstractions)
├── music_utils.py         # music21 MIDI utilities (zero TF dependency)
├── outputs.py             # Output helper
├── preprocess.py          # MIDI preprocessing
├── qa.py                  # Quality checks
├── test_utils.py          # Test stubs (all TF layer refs set to None)
├── data/                  # Training MIDI + sample audio (MP3)
│   ├── original_metheny.mid
│   ├── deepjazz_on_metheny...128_epochs.midi
│   ├── training_example.mp3
│   ├── 30s_seq.mp3
│   └── 30s_trained_model.mp3
└── images/                # Notebook diagrams
```

---

## Key dependencies (pinned)

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.3.1 | GPU support: MPS (Apple Silicon), CUDA, or CPU |
| torchaudio | 2.3.1 | Matched to torch version |
| numpy | 1.26.4 | Compatible with torch 2.3 |
| music21 | 9.1.0 | Grammar/MIDI processing |
| mido | 1.3.2 | Low-level MIDI I/O |

**Zero TensorFlow dependency** — this version has no TF imports anywhere.

---

## Architecture

The model is a **shared-weight LSTM** implemented as a native `nn.Module`:

```python
class DJModel(nn.Module):
    def __init__(self, n_values, n_a):
        self.lstm_cell = nn.LSTMCell(n_values, n_a)   # shared weights
        self.linear    = nn.Linear(n_a, n_values)      # output projection

    def forward(self, X, a0, c0):
        # unroll over Tx time steps
        for t in range(X.shape[1]):
            a, c = self.lstm_cell(X[:, t, :], (a, c))
            outputs.append(softmax(linear(a)))

    def generate(self, x0, a0, c0, Ty=50):
        # autoregressive: ŷ(t) → x(t+1)
```

- **Input**: One-hot tokens, dimension `n_values=91`
- **Hidden state**: 64-dimensional (`n_a=64`)
- **Training loss**: Negative log-likelihood averaged over time steps
- **Inference bridge**: `PTInferenceAdapter` wraps the model to match the music21 pipeline interface

---

## Author

Carlos Morillo — [github.com/morillo](https://github.com/morillo)
