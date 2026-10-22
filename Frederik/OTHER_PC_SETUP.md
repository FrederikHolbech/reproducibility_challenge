# Continue the CTNet A01 work on another PC

The repository contains the code, released checkpoint, results, and documentation. The destination computer only needs a compatible Python environment and the registered BCI Competition IV-2a data.

## 1. Get the repository

```powershell
git clone https://github.com/christianhst/reproducibility_challenge.git
Set-Location .\reproducibility_challenge\Frederik
```

If the repository is already present, run `git pull` and enter the `Frederik` directory.

## 2. Open it with Codex

Open the repository as the Codex workspace and use this handoff prompt:

```text
Continue my CTNet subject-A01 reproducibility challenge. Read
Frederik/README.md, Frederik/REPRODUCIBILITY_RECORD.md, and
Frederik/OTHER_PC_SETUP.md completely. Inspect the scripts and completed JSON
result. Do not start the 1,000-epoch run yet. First report the GPU model, CUDA
status, Python and PyTorch versions, whether the four required data files are
present, and an estimated runtime. Keep released-checkpoint evaluation,
strict retraining, and robustness experiments as separate results.
```

## 3. Create the environment

Python 3.10 is recommended because it matches the authors' stated environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
nvidia-smi
```

Install a PyTorch build compatible with the PC's NVIDIA driver and CUDA capability. The authors used PyTorch 1.13.1, but GPU installation commands differ by machine. Use the official PyTorch selector or ask Codex to determine the compatible command after inspecting `nvidia-smi`.

Then install the remaining packages:

```powershell
python -m pip install -r requirements.txt
```

The scripts support both the authors' older PyTorch serialization API and current PyTorch versions.

## 4. Obtain the data

Download BCI Competition IV dataset 2a from <https://www.bbci.de/competition/iv/> after completing any required registration. Download the official labels from the link in `data/README.md`.

Place the four files at:

```text
data/raw/A01T.gdf
data/raw/A01E.gdf
data/labels/A01T.mat
data/labels/A01E.mat
```

The data directory is excluded from Git to avoid redistributing registered competition data.

## 5. Build the cache and verify the checkpoint

```powershell
python .\reproduction\evaluate_a01_checkpoint.py --rebuild-cache
```

Expected result:

```text
correct: 262
n_test_trials: 288
accuracy_percent: 90.97222222222221
cohen_kappa: 0.8796296296296297
```

This creates `data/processed/A01.npz`. Subsequent evaluation can be run without `--rebuild-cache`.

## 6. Check the GPU before full training

```powershell
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Do not begin the long run unless CUDA reports `True`, unless a CPU run is intentional.

## 7. Run the full model later

Strict reproduction of the paper-era code path and the notebook's recorded A01 seed:

```powershell
python .\reproduction\train_a01.py --device cuda --epochs 1000 --seed 966 --validation legacy
```

The `legacy` mode intentionally preserves the paper-era overlapping validation implementation. It is needed for strict code reproduction, but it is not a recommended model-selection design.

Preserve these generated files and commit the small text results back to the `Frederik` folder:

```text
results/a01_retrained.json
results/a01_training_history.csv
results/a01_retrained_state.pt
```

Also record the GPU model, runtime, versions, selected epoch, accuracy, kappa, and confusion matrix.

