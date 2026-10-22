# Frederik — CTNet A01 reproducibility challenge

This folder contains the reproducibility work for CTNet subject A01. Everything needed to inspect the work, evaluate the authors' checkpoint, and later retrain the model is committed here except the BCI Competition IV-2a data.

## Current status

The released A01 checkpoint has been evaluated against the official A01E labels:

| Result | Value |
| --- | ---: |
| Correct trials | 262/288 |
| Accuracy | 90.9722% |
| Paper Table 2 | 90.97% |
| Cohen's kappa | 0.8796 |

This exactly reproduces the published accuracy after rounding. It is an artifact-level reproduction; the planned 1,000-epoch training-from-scratch run has not yet been performed.

## Folder contents

| Path | Purpose |
| --- | --- |
| `REPRODUCIBILITY_RECORD.md` | Detailed scientific record and findings |
| `OTHER_PC_SETUP.md` | Complete handoff instructions for another computer |
| `reproduction/evaluate_a01_checkpoint.py` | Preprocess data and evaluate the released checkpoint |
| `reproduction/train_a01.py` | Later full subject-A01 training run |
| `models/2a/model_1.pth` | Authors' released A01 checkpoint |
| `results/` | Completed metrics and trial-level predictions |
| `data/README.md` | Required data files, locations, and hashes |
| `requirements.txt` | Python dependencies other than PyTorch |

The repository root contains the authors' original CTNet source and notebooks. The scripts in this folder are deliberately self-contained so they can record the exact A01 protocol and reproducibility findings without changing the upstream files.

## Quick start

Use Python 3.10. Install a PyTorch build appropriate for the destination computer's NVIDIA driver, then install the remaining packages:

```powershell
python -m pip install -r requirements.txt
```

Obtain the registered BCI Competition IV-2a files described in `data/README.md`, then build the cache and evaluate:

```powershell
python .\reproduction\evaluate_a01_checkpoint.py --rebuild-cache
```

Once `data/processed/A01.npz` exists, neither evaluation nor training needs to parse the raw GDF files again.

The later strict training command is:

```powershell
python .\reproduction\train_a01.py --device cuda --epochs 1000 --seed 966 --validation legacy
```

Do not start that long run until CUDA availability and the environment have been checked. See `OTHER_PC_SETUP.md` for the complete procedure.

## Provenance

- Paper: <https://doi.org/10.1038/s41598-024-71118-7>
- Authors' repository: <https://github.com/snailpt/CTNet>
- Upstream revision examined: `fb83814abc55bafaea6e71cc18475bf331c8d28b`
- Course repository: <https://github.com/christianhst/reproducibility_challenge>

