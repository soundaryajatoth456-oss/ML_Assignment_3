# Machine Learning Assignment 3

Soundarya, roll `m24ma2004`. The split seed is `242004`, the digits of that roll number in order.

Dataset: CIFAR-10. One stratified split of the official 50,000 training images into 45,000 train and 5,000 validation. The official test set is not used for tuning.

Q1–Q4 have been run. The rules are in `configs/q1_fairness.json`, `configs/q2_plan.json`, `configs/q3_plan.json` and `configs/q4_plan.json`. Those files were written before the corresponding runs. Do not edit them to match a result.

## Environment

The cluster module `python/3.10.pytorch` provides PyTorch 2.1.2 with CUDA 11.8. Its bundled torchvision was built for CUDA 12.1 and does not import, so torchvision 0.16.2 is installed into a local virtual environment that can still see the module’s PyTorch.

From this directory:

```bash
module load python/3.10.pytorch
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

There is no GPU on the login node (`nvidia-smi` is absent and `torch.cuda.is_available()` is false). Phase 1 does not need one. Later training should be submitted with `srun` on the `mtech` partition, for example `srun --partition=mtech --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=04:00:00 --pty bash`.

## Phase 1 — data

```bash
PYTHONNOUSERSITE=1 .venv/bin/python scripts/prepare_data.py
```

`PYTHONNOUSERSITE=1` keeps packages installed in the home directory off `sys.path`. Without it, an unrelated local torchvision can be imported ahead of the one in `.venv`.

The same steps can be read in `notebooks/01_dataset.ipynb`. That notebook calls `src/common/data.py`; it does not define a second split.

What the command writes:

- `splits/split_seed26003.npz` — sorted train and validation indices
- `splits/norm_stats_seed26003.npz` (and `.json`) — per-channel mean and population std of the training split only
- `splits/verification_seed26003.json` — class counts, overlap check, index hash
- `splits/figure_example_idx_seed26003.npy` — which training images the sample grid draws
- `figures/class_samples.png`, `class_distribution.png`, `mean_images.png`, `pixel_statistics.png`, plus PDF copies
- `figures/dataset_description.md`
- `results/phase1/class_counts.csv`, `train_per_class_pixel_stats.csv`, `environment.json`

Running the command again reloads the split and checks that a fresh draw from seed 26003 matches it. It does not replace the file.

There is no git commit yet, so there is no commit hash for page 1 of the report. The repository has not been published, so the report leaves the GitHub link as "not yet published".

## Two notebooks

`notebooks/Assignment_3_Part_1.ipynb` covers the split, Question 1 (matched comparison and label noise) and Question 2 (clean reuse, spatial permutation, grayscale). `notebooks/Assignment_3_Part_2.ipynb` covers Question 3 (autoencoders, denoising, linear probes) and Question 4 (SVM sweep, hard pairs, clustering).

Each notebook finds the project directory itself, imports `src/`, reloads the saved split, and reads the logs under `results/`. `RECOMPUTE` is `False`. Run All does not train. Set it to `True` only on a GPU node when the logs are missing; a finished json is reloaded rather than replaced. The older notebooks `01_dataset.ipynb` through `05_q4.ipynb` are the per-question notes and are not required by these two.

From this directory, with the module and the virtual environment above:

```bash
module load python/3.10.pytorch
export PYTHONNOUSERSITE=1
cd /scratch/m26csa003/ML_assignment_3
.venv/bin/python -m pip install -r requirements-notebooks.txt
.venv/bin/python -m ipykernel install --user --name ml-a3 --display-name "Python (ML assignment 3)"
.venv/bin/python scripts/execute_notebook.py notebooks/Assignment_3_Part_1.ipynb
.venv/bin/python scripts/execute_notebook.py notebooks/Assignment_3_Part_2.ipynb
```

Run the two `execute_notebook.py` commands as separate processes. Neither notebook reads variables from the other. `requirements-notebooks.txt` lists the notebook packages used to execute them. They are not part of the training environment in `requirements.txt`.

## Report

The CVPR source is `report/main.tex` with `report/references.bib`, `report/cvpr.sty` and `report/ieeenat_fullname.bst`. Figures used by the paper are copied into `report/figures/` from `figures/`. Compile from `report/`:

```bash
cd /scratch/m26csa003/ML_assignment_3/report
tectonic main.tex
```

`tectonic` is not on this cluster's `PATH`. The PDF in the tree was built with Tectonic 0.15.0, which runs XeTeX. The `times` package loaded by `cvpr.sty` has no Unicode font under XeTeX, so `main.tex` sets the body in Nimbus Roman, the Times-compatible face that package selects. Those four files are in `report/fonts/` with their licence (`LICENSE`, `COPYING`): GNU AGPL-3.0 with an exception that allows embedding the face in the PDF. The style file is the official CVPR author kit (`cvpr-org/author-kit`, package dated 2027). The PDF is `report/main.pdf`. It is six pages, including the references.

## Where the numbers are

| Question | Tables and logs | Figures |
| --- | --- | --- |
| Split | `splits/verification_seed26003.json` | `figures/class_samples.png` and the other Phase 1 figures |
| Q1 | `results/q1/q1a_model_table.csv`, `results/q1/test_results.json`, `results/q1/q1b_summary.csv` | `figures/q1/` |
| Q2 | `results/q2/q2_summary.csv`, `results/q2/test_results.json` | `figures/q2/` |
| Q3 | `results/q3/q3a_selection.csv`, `results/q3/q3b_classification.csv`, `results/q3/q3c_representation.csv` | `figures/q3/` |
| Q4 | `results/q4/svm_sweep.csv`, `results/q4/svm_test.json`, `results/q4/hard_pairs.json`, `results/q4/clustering.json` | `figures/q4/` |

`REPORT_NOTES.md` maps each reported number to the file that produced it. `EXPLAINER.md` is the short walkthrough of the same measurements.

## Q1

The comparison rules were written to `configs/q1_fairness.json` before training. Do not edit that file to match a result.

Smoke check (CPU, not a reported number):

```bash
PYTHONNOUSERSITE=1 .venv/bin/python scripts/q1_smoke.py
```

Full run on one GPU. A finished probe, final model, or noise fraction is reloaded from its json, so a second submission continues where the last one stopped:

```bash
sbatch --partition=medium --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 \
  --job-name=q1_m26csa003 \
  --output=results/q1/train.log --error=results/q1/train.log scripts/q1_srun.sh
```

The Q1b rerun that wrote `results/q1/q1b/` was logged in `results/q1/q1b.log`.

Figures can be redrawn from the logs:

```bash
PYTHONNOUSERSITE=1 .venv/bin/python scripts/q1_figures.py
```

The same design is written up in `notebooks/02_q1.ipynb`. The measured reading is in `EXPLAINER.md` and `REPORT_NOTES.md`.

## Q2, Q3 and Q4

Smoke check (CPU, not a reported number):

```bash
PYTHONNOUSERSITE=1 .venv/bin/python scripts/q234_smoke.py
```

Full run on one GPU. Each question reloads a finished stage from its json, so a second submission continues:

```bash
sbatch --partition=medium --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=08:00:00 \
  --job-name=q234_m26csa003 \
  --output=results/q234.log --error=results/q234.log scripts/q234_srun.sh
```

Redraw the figures:

```bash
PYTHONNOUSERSITE=1 .venv/bin/python scripts/q2_figures.py
PYTHONNOUSERSITE=1 .venv/bin/python scripts/q3_figures.py
PYTHONNOUSERSITE=1 .venv/bin/python scripts/q4_figures.py
```

`notebooks/03_q2.ipynb`, `notebooks/04_q3.ipynb` and `notebooks/05_q4.ipynb` read the plan files. They do not choose a new split or a new hyperparameter.
