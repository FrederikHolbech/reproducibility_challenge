# CTNet A01 reproducibility record

This document records the completed work for the Seminars in Data Science reproducibility challenge on CTNet. The assigned target is the subject-specific classification accuracy for subject A01 in Table 2 of Zhao et al. The released checkpoint reproduces the reported result exactly after rounding. A full training-from-scratch run has deliberately been left for a later GPU session.

## Assignment and deliverable

- Paper: Wei Zhao, Xiaolu Jiang, Baocan Zhang, Shixiao Xiao, and Sujun Weng, “CTNet: a convolutional transformer network for EEG-based motor imagery classification,” *Scientific Reports* 14, 20237 (2024).
- Paper DOI: <https://doi.org/10.1038/s41598-024-71118-7>
- Authors’ repository: <https://github.com/snailpt/CTNet>
- Dataset: BCI Competition IV, dataset 2a: <https://www.bbci.de/competition/iv/>
- Assigned result: reproduce the Table 2 CTNet classification accuracy for subject A01.
- Course report: at most six pages, including a one-page summary. The course slides give a submission deadline of 27 November 2026 at 12:00.

## Exact target

Table 2 contains subject-specific results for BCI Competition IV-2a. This is not the cross-subject A01 result in Table 4.

| Item | Target |
| --- | --- |
| Dataset | BCI Competition IV-2a |
| Subject | A01 |
| Protocol | Subject-specific |
| Training recording | A01T |
| Evaluation recording | A01E |
| Number of evaluation trials | 288 |
| Paper accuracy | 90.97% |
| Exact corresponding count | 262/288 = 90.972222...% |

The main claim supported by this result is that CTNet performs strongly in subject-specific motor-imagery classification. For A01, CTNet is the best method listed in Table 2.

## Completed checkpoint evaluation

The authors’ released `CTNet/models/2a/model_1.pth` checkpoint was evaluated on the official A01E trials after reproducing the repository’s preprocessing and normalization.

| Evaluation | Correct | Total | Accuracy | Cohen’s kappa | Difference from paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released A01 checkpoint | 262 | 288 | 90.9722% | 0.8796 | 0.0022 percentage points before rounding |

The result therefore matches the paper’s 90.97% after rounding to two decimals.

Confusion matrix, with rows representing true labels and columns predicted labels:

| True class | Left hand | Right hand | Feet | Tongue |
| --- | ---: | ---: | ---: | ---: |
| Left hand | 69 | 2 | 1 | 0 |
| Right hand | 5 | 66 | 1 | 0 |
| Feet | 1 | 0 | 59 | 12 |
| Tongue | 0 | 0 | 4 | 68 |

This is an artifact-level reproduction: it establishes that the released preprocessing pipeline, trained model, and evaluation data produce the published A01 number. It is not yet an independent reproduction of model training.

## Preprocessing reproduced

The following choices match the authors’ code:

1. Read the official A01T and A01E GDF recordings.
2. Keep 22 EEG channels and discard the three EOG channels.
3. Use the four motor-imagery classes: left hand, right hand, feet, and tongue.
4. Extract each trial from 0 to 3.996 seconds, producing 1,000 samples at 250 Hz.
5. Apply no extra filtering, baseline correction, or artifact removal.
6. Compute one scalar mean and standard deviation from A01T.
7. Standardize both A01T and A01E with those A01T statistics.
8. Evaluate the model on the 288 official A01E labels.

The measured A01T statistics were:

- Mean: `3.6949405862827505e-07`
- Standard deviation: `1.0187522452967678e-05`

## Software environment used for checkpoint evaluation

- Python 3.13.0
- PyTorch 2.10.0 CPU
- NumPy 2.2.1
- MNE 1.13.2

The repository requests Python 3.10, PyTorch 1.13.1, and MNE 1.5.1. MNE 1.5.1 was initially installed, but its GDF reader fails with NumPy 2 because of a documented integer-overflow incompatibility. MNE 1.13.2 was therefore used only to read the GDF files. Epoch definitions, channel selection, labels, normalization, model weights, and metric computation were not changed. The relevant MNE issue is <https://github.com/mne-tools/mne-python/issues/12907>.

## Repository findings that must be reported

### The saved notebook agrees on A01 but not on the overall mean

The output embedded in `CTNet_2a_82.91.ipynb` records the following A01 run:

- Seed: 966
- Selected checkpoint epoch: 979
- Test accuracy: 0.9097222222
- Cohen’s kappa: 0.8796296296

This is the same A01 result as Table 2. However, the notebook’s nine-subject mean is 82.90895%, while the paper reports 82.52%. The assigned A01 result is unaffected, but the discrepancy should be disclosed rather than silently ignored.

### The 2024 validation implementation does not match the paper description

The paper says that 30% of the initial training set was reserved as validation data. The paper-era code instead creates a new split inside every shuffled mini-batch and every epoch.

For a raw batch of 72 trials, `number_validate = int(0.3 * 72) = 21`. The 2024 notebook uses:

```python
train_data = img[:-number_validate]       # positions 1–51
val_data_list.append(img[number_validate:])  # positions 22–72
```

Consequences:

- Training receives 51 of 72 raw trials.
- Validation also receives 51 of 72 trials, or 70.8%, rather than 30%.
- Positions 22–51, 30 trials per batch, occur in both training and validation in the same epoch.
- The split changes at every epoch because the data loader reshuffles the full training set.
- Validation loss is therefore not measured on a fixed independent holdout.

The current repository changed the validation slice in April 2025 to `img[-number_validate:]`, removing the within-epoch overlap. It still reshuffles and reconstructs the validation subset every epoch, so it still is not the fixed holdout described in the paper.

A strict reproduction of the published code should preserve the legacy behavior and label it clearly. A robustness experiment should instead use one fixed, stratified 70/30 split created before training.

### Seed selection is not reproducible by default

The repository selects each subject’s seed using `np.random.randint(2024)` before establishing a fixed top-level NumPy seed. A fresh execution is therefore not guaranteed to choose the notebook’s A01 seed of 966. The prepared training script sets seed 966 explicitly.

### Batch-size interpretation is unclear

The paper states a subject-specific batch size of 288. The code begins with a raw batch size of 72 and creates 216 augmented trials per batch when `N_AUG=3`. Before validation removal, this gives 288 examples. With the legacy split, the actual optimization batch contains 51 real plus 216 augmented trials, or 267 examples.

## Local files

| Path | Purpose |
| --- | --- |
| `reproduction/evaluate_a01_checkpoint.py` | Preprocessing and released-checkpoint evaluation |
| `reproduction/train_a01.py` | A01 training from scratch with explicit `legacy` and `corrected` validation modes |
| `results/a01_checkpoint.json` | Exact metrics, versions, hashes, and confusion matrix |
| `results/a01_predictions.csv` | Trial-level true labels and predictions |
| `data/processed/A01.npz` | Cached A01T/A01E arrays and labels |
| `models/2a/model_1.pth` | Authors’ released A01 checkpoint |

The cloned CTNet revision used for this work is:

```text
fb83814abc55bafaea6e71cc18475bf331c8d28b
```

## Commands

Re-evaluate the released checkpoint:

```powershell
python .\reproduction\evaluate_a01_checkpoint.py
```

Run the later full 1,000-epoch reproduction on a CUDA machine using the recorded seed and paper-era validation behavior:

```powershell
python .\reproduction\train_a01.py --device cuda --epochs 1000 --seed 966 --validation legacy
```

Run the repository’s April 2025 non-overlapping validation slice for comparison:

```powershell
python .\reproduction\train_a01.py --device cuda --epochs 1000 --seed 966 --validation corrected
```

The `corrected` option only removes within-epoch overlap. It does not create a fixed validation set.

## Later full-training plan

1. Use a CUDA GPU and an environment close to Python 3.10 and PyTorch 1.13.1.
2. First run seed 966 with `--validation legacy` as the strict code reproduction.
3. Preserve the resulting JSON, history CSV, state dictionary, console log, runtime, GPU model, and software versions.
4. Run at least three additional predetermined seeds to estimate variation rather than relying on one favorable run.
5. Repeat with a fixed stratified holdout as a robustness experiment. This requires a further code change; the current `corrected` mode is not sufficient.
6. Present checkpoint evaluation, strict retraining, and robustness experiments as separate results.
7. Discuss negative or different results rather than tuning until 90.97% is reached.

On the current CPU-only computer, a two-epoch smoke test took approximately eight seconds of training time. A full run is estimated at roughly 65–70 minutes, so the GPU machine is preferable.

## Current conclusion

The published A01 number is reproducible from the authors’ released artifact: the measured value is 262/288 = 90.9722%, which rounds to the reported 90.97%. Independent training reproducibility remains open until the planned 1,000-epoch GPU run is completed. The validation overlap, changing validation membership, unseeded seed selection, notebook/paper mean discrepancy, and batch-size interpretation are substantive reproducibility findings and should be included in the course report.


