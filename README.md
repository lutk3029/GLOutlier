# GLOutlier: Unifying Global and Local Anomaly Detection for Time Series

Code and experimental artifacts for **“Unifying Global and Local Anomaly
Detection for Time Series”** by Tongkai Lu, Shuai Ma, and Zhongxi Zhang.

## Reproduce the 48-dataset results and paper aggregates

The repository contains the per-dataset Precision, Recall, F1, and runtime values
behind Appendix Tables A1–A6, Table 6, Table B7, and the Bonferroni–Dunn diagram:

- [`results/timeeval_per_dataset.csv`](results/timeeval_per_dataset.csv): 672 rows
  (48 datasets × 14 methods), transcribed from Appendix Tables A1–A6.
- [`experiments/timeeval_48.csv`](experiments/timeeval_48.csv): the exact
  23 SMD + 16 SVDB + 4 MITDB + 2 Exathlon + 3 Daphnet selection and its TimeEval
  download links.
- [`experiments/build_timeeval_results.py`](experiments/build_timeeval_results.py): computes
  Table 6, the tie-corrected Friedman test, Holm post-hoc comparisons in Table B7,
  and the Bonferroni–Dunn diagram from the per-dataset CSV.

Run:

```sh
python3 experiments/build_timeeval_results.py
```

This writes [`experiment_results/table6.csv`](experiment_results/table6.csv),
[`experiment_results/table6_exact.csv`](experiment_results/table6_exact.csv),
[`experiment_results/table_b7.csv`](experiment_results/table_b7.csv),
[`experiment_results/statistical_summary.txt`](experiment_results/statistical_summary.txt), and
[`experiment_results/figure_b1_cd_diagram.svg`](experiment_results/figure_b1_cd_diagram.svg).
The script uses only the Python standard library and validates the 48 × 14 input
shape before computing any aggregate.

To audit the transcription against the submitted PDF itself:

```sh
pdftotext -layout \
  "KAIS2025_Unifying_Global_and_Local_Anomaly_Detection_for_Time_Series__v3_ submitted(1).pdf" \
  paper.txt
python3 experiments/extract_timeeval_results.py \
  paper.txt results/timeeval_per_dataset.csv
python3 experiments/build_timeeval_results.py
```

The appendix reports per-dataset scores to one decimal place. Consequently, a
source mean exactly on an `x.x5` boundary can differ by 0.1 depending on the
rounding convention; the scripts always recompute from the committed displayed
values and retain two decimals for the overall 48-dataset mean.

## Run GLOutlier on exactly the same 48 TimeEval datasets

**This is the batch entry point for our method:**

```sh
python3 experiments/run_timeeval_experiments.py \
  --dataset-root /path/to/extracted/timeeval-datasets \
  --device cuda
```

Use `--dry-run` to inspect all resolved commands, or select one sequence while
checking the environment:

```sh
python3 experiments/run_timeeval_experiments.py \
  --dataset-root /path/to/extracted/timeeval-datasets \
  --only SMD/machine-1-1 --device cpu
```

The batch script invokes [`run_timeeval.py`](run_timeeval.py) for each manifest
row. The single-dataset runner executes dataset-generic discretization,
local-factor partitioning, association-rule global detection with learned
tolerances, AT-LSH local detection, and unification. Each
run writes `predictions.csv`, `metrics.json`, intermediate AT-LSH data, and its
checkpoint under `timeeval_runs/<collection>/<dataset>/`.

The five official TimeEval archives are linked directly below; they are not
copied into this branch:

- [SMD (99 MB)](https://my.hidrive.com/api/sharelink/download?id=W0CGA01i)
- [SVDB (103 MB)](https://my.hidrive.com/api/sharelink/download?id=lmCmAjUP)
- [MITDB (176 MB)](https://my.hidrive.com/api/sharelink/download?id=YcCmAEXy)
- [Exathlon (106 MB)](https://my.hidrive.com/api/sharelink/download?id=q9imgqn3)
- [Daphnet (15 MB)](https://my.hidrive.com/api/sharelink/download?id=SfCmg30B)

These links point to the preprocessed TimeEval canonical CSV archives described
on the [official TimeEval dataset page](https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html).
Do not run every sequence in those five collections: the paper uses only the 48
manifest rows. TimeEval labels SMD as **semi-supervised**, so the runner uses each
normal `.train.csv`. TimeEval labels SVDB, MITDB, Exathlon, and Daphnet as
**unsupervised** and does not provide normal training files for them; for those
collections the runner fits on the test series values without reading
`is_anomaly`, then reads that label column only to calculate the final point-wise
metrics. No label-dependent point adjustment is applied.

All 13 comparison implementations, upstream links, input conversions, and run
commands are documented in
[`experiments/baseline_commands.md`](experiments/baseline_commands.md). They remain in
their original repositories and are intentionally not downloaded into this
branch.

## Environment

The original experiments used Ubuntu 18.04, Python 3.7.13, PyTorch
1.0.1.post2, CUDA 9.0, an Intel Xeon Gold 6148 CPU, and an NVIDIA Tesla V100
32 GB GPU. Install the recorded dependencies with:

```sh
pip install -r requirements.txt
```

The paper settings used by the TimeEval runner are local factor 0.5, rule support
0.7, rule confidence 0.9, a 100-point AT-LSH window, and a threshold at the 99th
percentile of training anomaly scores. Tables 5, 6, and Appendix A use ordinary
point-wise Precision, Recall, and F1 without point adjustment.

## Original SWaT workflow

Download SWaT and WADI from [iTrust](https://itrust.sutd.edu.sg/itrust-labs_datasets/).
The original processed data are also available from the authors as
[processed files](https://drive.google.com/drive/folders/1USOqY4xu4_kZTxM2_784TowQ2voUftQX?usp=sharing)
and [pickle files](https://drive.google.com/drive/folders/1ESdEykcOPRwwwy6lFV2_38TS_yqcTR51?usp=drive_link).
Place those under `data/` and `file/`, respectively.

Run the combined workflow with `python run.py`, or execute its stages directly:

```sh
python data_discretization.py
python data_partition.py
python AR/All.py
bash ATR/train.sh
bash ATR/test.sh
python unidfying_detection_result.py
```

Discretization and partitioning must finish before detection. The original SWaT
scripts use a local factor of 0.5 and contain the SWaT sensor/actuator selection
used for Table 5; the dataset-generic TimeEval workflow is the separate batch
entry point highlighted above.
