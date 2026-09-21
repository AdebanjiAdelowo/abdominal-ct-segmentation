# Evaluation protocol v1 (for a new model; not the historical result)

Status: **implemented and tested on synthetic data only. It has not been run on the real MSD Task03 data and no result from it exists.** The frozen split file has not been generated yet because it needs the dataset listing.

## 0. What this is and is not

The historical model reports *0.9886 Dice on the 26-volume 128³ centre-cropped validation split used for model selection.* That is not independent test performance and not full-volume performance. It cannot be given an independent test set: it trained on 105 of the 131 labelled cases and the other 26 chose its checkpoint. Its HD95 was logged with an invalid metric and could only be in voxels; its `best.pth` (epoch 191, present on 2026-09-15) is no longer available; a corrected physical-unit HD95 cannot be recovered.

This protocol defines how a **new** model is trained and evaluated. Its numbers are a separate experiment. They must never replace, and are not evidence about, the historical result.

## 1. Frozen split

| Split | Cases (of 131) | Role |
|---|---|---|
| train | 85 | gradient updates |
| val | 20 | checkpoint selection, early stopping, any development decision |
| test | 26 | one final evaluation of one frozen checkpoint |

### 1.1 Construction (seed 42, patient-level, deterministic)

1. **Historical validation cases.** Identify the exact 26 cases that chose the historical checkpoint by applying the historical split logic (`src.data.dataset.historical_train_val_split`: `np.random.default_rng(42).permutation` of the sorted case names, first `max(1, int(n × 0.2))`) to the historical case listing. The same function backs the historical loader, so the two cannot drift apart.
2. **Final-test eligibility.** `eligible_test = all_cases − historical_validation_cases` (105 cases).
3. **Final test.** Draw 26 cases from `eligible_test` with `np.random.default_rng(42)`. The test set is drawn first, so its membership does not depend on the validation fraction.
4. **Train and validation.** The remaining 105 cases (which include all 26 historical validation cases) are divided into 85 train and 20 validation with the same generator.

Historical validation cases may appear in the new train or validation sets. They can never appear in the new test set. This is an experimental-design constraint, not seed searching: the seed is fixed at 42, the CLI has no `--seed` option, and no alternative split is generated or compared.

### 1.2 Why the historical validation cases are excluded from the final test

Those 26 cases were already used to evaluate the historical model during training and to select its checkpoint, and the new model inherits that model's hyperparameters. Excluding them means no case in the new final test set has ever been used to evaluate or select any model of this project, which gives the new experiment a genuinely untouched final test set. (A literal seed-42 split of all 131 cases would have made the new test set identical to the historical validation set.)

### 1.3 The historical listing is not guessed

The historical run used `.npy` files from a Kaggle mirror that is not available here. By default the historical listing is taken to be the MSD `imagesTr` listing, and generation is **refused** unless that listing passes `check_historical_naming`: every name is `liver_<id>`, and the first three files sorted by full filename (as the historical overlay code sorted them) are `liver_0`, `liver_100`, `liver_101`. This is a necessary consistency check, not proof of identity; the identity of the two listings remains an assumption that cannot be verified without the mirror. If the real naming differs, `make` stops, and the historical listing must be supplied explicitly (`--historical-listing`, one name per line). Every historical validation id must exist in the dataset, otherwise generation fails instead of guessing.

### 1.4 Freezing, metadata and fingerprint

* The split is written **once** to `splits/msd_task03_v1.json`, committed **before** any training; overwriting is refused; the CLI refuses a dataset that is not exactly 131 cases. A different seed or fractions is a new protocol version with a new file and new results.
* **The new test set does not exist yet.** The split file can only be generated from the real MSD Task03 case listing, which is not available in this repository. Nothing here should be read as a set of test cases until that file has been generated and committed.
* The file records: seed; the historical validation case ids; the derivation method; the exclusion rule; the final-test eligibility population; the train, validation and test lists; counts; how many historical validation cases fall in the new train and validation sets; the listing source; and the fingerprint.
* **Fingerprint:** SHA-256 over the canonical content of the three case lists **and** the construction (historical validation ids, exclusion rule id, seed). Moving a case, editing the historical ids or changing the seed invalidates it. Loading also re-checks disjointness, that no historical validation case is in the test set, and that the eligibility list equals `all_cases − historical_validation`. The fingerprint is stored in every checkpoint (twice) and every result.
* Before freezing, `python -m src.data.splits describe` may be used to look at non-performance characteristics (spacing, liver volume) for pathological imbalance. It reads headers and labels only and is not a search over seeds.
* MSD Task03 pools several institutions and the public metadata does not label the site, so the split is random, not site-stratified. Site effects are a stated limitation.

## 2. What may use which split

| Activity | train | val | test |
|---|---|---|---|
| fit weights | yes | no | never |
| choose `best.pth` (patch-level val Dice) | no | yes | never |
| early stopping, LR / architecture / loss decisions | no | yes | never |
| final reported evaluation | no | reported and labelled as validation | once |

Enforced in code:

* `protocol_train` loads the split with `include_test=False`; the test case names and the test-eligibility list are removed before training code sees them. It refuses to start unless the split file is committed and unmodified and the working tree has no uncommitted changes to tracked files (`--allow-dirty` for experiments only), so the recorded git commit and fingerprint describe what ran. The training config records no case identifiers.
* The checkpoint stores the fingerprint, seed, `selection_split: val`, git commit and dirty flag, timestamp and model configuration. `evaluate_protocol` refuses a checkpoint that lacks them, was trained on a different split file, or does not record validation-based selection. The historical checkpoint therefore cannot be scored under this protocol.
* **Test evaluation guard.** `--split test` requires (a) `--confirm-final-test`; (b) a validation evaluation of the *same checkpoint hash* in the same output directory; (c) a lock file stored **beside the split file** (`<splits>.test_lock.json`, or `--lock-file`, e.g. on Drive), not in the output directory, so a fresh output directory does not bypass it. The first test evaluation records the checkpoint hash; a different checkpoint is refused; every attempt is appended. This is an audit trail against accidental reuse, not cryptography. Choosing among models on test results invalidates the test set; start a new protocol version with a new split instead.

## 3. Geometry and units

* Volumes are read from the original NIfTI files on their native grid, reoriented to RAS+ (`nibabel.as_closest_canonical`). No resampling.
* Voxel spacing comes from the reoriented affine (column norms), so `spacing[i]` always belongs to array axis `i` whatever the on-disk orientation.
* Sheared or oblique grids are refused (`index × spacing` would not be a physical distance). Image and label must agree in shape, spacing and affine or loading fails.
* HD95 uses those spacings: a **physical surface distance in millimetres**.

## 4. Full-volume evaluation

* Each case is reconstructed in full by MONAI sliding-window inference (patch size, overlap, Gaussian blending from the checkpoint's own config). Axes shorter than the patch are reflect-padded as in training, and the prediction is cropped back.
* Preprocessing is the shared `preprocess_ct` (HU clip [-200, 250], per-volume z-score), the same function training uses.
* Target: every positive label (liver and tumour) is foreground, as in the historical pipeline.
* This is **not** the 128³ centre-crop score. The per-epoch `val_hd95` in training logs is a voxel-unit diagnostic on one patch and must not be reported.

## 5. Metrics, aggregation and complete failures

* Per case: Dice, and HD95 = max of the directed 95th percentiles of surface-to-surface distances (prediction to reference and reference to prediction), surfaces from binary erosion, in mm. Also predicted and reference volume (mL) for sanity checks and a per-case `status`.
* Aggregate per split: unweighted mean over cases (each patient counts once; macro average), sample standard deviation (ddof = 1), median, IQR, min/max, and a seeded 95% percentile-bootstrap CI of the mean over cases.
* **Complete failure = empty prediction.** HD95 is undefined (stored as infinity, status `empty_prediction`). Dice keeps the case (Dice = 0). The summary always reports: total cases; cases with finite HD95; number of complete failures and their ids; Dice over all cases; and HD95 statistics under the key `finite_cases_only`. The generated `report_text` puts "COMPLETE FAILURES: k of n" before any finite-case HD95 and never presents a finite-case mean as if it covered all patients. An empty *reference* label is treated as a data error and raises.
* Outputs: `<split>_per_case.csv`, `<split>_summary.json` (with checkpoint SHA-256, split fingerprint, git commit, timestamp, aggregation text).

## 6. Checkpoint persistence

Trained checkpoints are expensive and this project has already lost one.

* `best.pth` and `last.pth` are written atomically (temp file, fsync, rename) and carry: epoch, model / optimiser / scheduler / AMP-scaler state, validation Dice used for selection, best Dice so far, seed, split fingerprint, full config including the model architecture, git commit and dirty flag, UTC timestamp, format version. `last.pth` is written after the scheduler step so resume continues the schedule exactly.
* `checkpoint_manifest.json` records epoch, Dice, timestamp and the SHA-256 of `best.pth`.
* With `--mirror-dir` (e.g. Google Drive), `best.pth` is copied on every improvement, `last.pth` every 5 epochs and at the end, plus the manifest and metrics log. Each copy is written under a temp name, re-hashed, and only then renamed into place. `run_config.yaml` and the split file are stored alongside.
* `--resume last.pth` continues training (refused if the checkpoint's split fingerprint differs) and appends to the metrics log. Resume is not bit-exact because RNG state is not restored.
* Checkpoints and the dataset are never committed to Git.

## 7. Smoke test (infrastructure only)

`python -m src.smoke_test` runs, on about 4 training and 2 validation cases for 2 epochs, the same code paths as a real run: NIfTI loading and geometry, training, checkpoint save, mirror, reload with fingerprint verification, resume check, full-volume inference, Dice, mm-HD95, and serialisation. It never loads the test case names. Its outputs carry a `SMOKE_TEST` stamp and a `NOT_A_RESULT.txt` file. Its metrics come from a barely trained model and must never be reported. `--synthetic` runs it on a generated dataset with no MSD data.

## 8. Running it

`colab/run_protocol.ipynb` is a thin launcher that only calls the commands below (nothing is reimplemented in cells): pin the commit, mount Drive, validate the dataset, verify or generate the split, smoke test, train with a Drive mirror, evaluate validation, and (only with an explicit flag) run the final test evaluation.

```bash
python -m src.data.validate_dataset --data-dir Task03_Liver --expect-cases 131
python -m src.data.splits make --data-dir Task03_Liver --out splits/msd_task03_v1.json   # once; then commit it
python -m src.smoke_test --data-dir Task03_Liver --splits splits/msd_task03_v1.json --out-dir runs/smoke
python -m src.training.protocol_train --data-dir Task03_Liver --splits splits/msd_task03_v1.json \
    --out-dir runs/protocol_v1 --mirror-dir /persistent/protocol_v1 --seed 0
python -m src.inference.evaluate_protocol --checkpoint /persistent/protocol_v1/best.pth \
    --data-dir Task03_Liver --splits splits/msd_task03_v1.json --split val --out-dir runs/eval
# only when every modelling decision is final:
python -m src.inference.evaluate_protocol ... --split test --confirm-final-test
```

## 9. Reporting rules

* Report validation and test separately and label them; label the model as new, with seed, split fingerprint and commit; state it is one training run.
* Do not describe validation-split numbers as test performance, and do not present the new result as a reproduction or improvement of the historical 0.9886. A lower score under the stronger protocol is an acceptable outcome and must not be hidden.
* The case-bootstrap CI covers test-case sampling only, not training variance.
* Public documents should be updated only from the saved result files, and only after the run has completed.

## 10. What is verified and what is not

Verified by the test suite (synthetic data and an assumed `liver_0`..`liver_130` listing): split sizes 85/20/26, union = all cases, mutual disjointness, `test ∩ historical validation = ∅` against an independent re-implementation of the historical split, determinism, fingerprint sensitivity to membership and construction, rejection of a historical case in the test set even with a recomputed fingerprint, write-once, seed 42, refusal to guess the historical listing, dataset-size refusal, and the training view containing no test names; spacing follows array axes under axis permutation and flips; sheared grids and image/label mismatches refused; HD95 equals shift × spacing per axis under anisotropic spacing (3.0, 0.7, 0.9 mm) and matches an independent distance-transform oracle; the full NIfTI to sliding-window to mm pipeline; empty-prediction failures visible in the summary; training loaders exclude test cases; checkpoint completeness, atomicity, verified mirroring and resume; leakage and lock guards including a fresh output directory; the smoke test. Re-introducing a voxel-space, default-spacing or unpermuted-spacing bug makes the suite fail.

Not verified: behaviour on real MSD volumes (memory, runtime, orientation quirks), execution of the Colab notebook, and any accuracy claim.
