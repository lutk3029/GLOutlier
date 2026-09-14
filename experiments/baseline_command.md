# Baseline code and invocation

The baseline repositories are intentionally **not vendored in this branch**. Clone
each upstream project into a separate environment, convert the selected TimeEval
CSV to that project's documented input layout, and run one dataset at a time. The
comparison in the paper uses point-wise Precision/Recall/F1 without point
adjustment. Except for the settings listed below, use the upstream/default model
and training parameters.

TimeEval CSV files have an index in the first column, feature channels in the
middle, and `is_anomaly` in the last column. The exact 48-file selection is in
[`timeeval_48.csv`](timeeval_48.csv). SMD supplies normal training files. Exathlon,
SVDB, MITDB, and Daphnet are TimeEval `unsupervised` collections; fit those
methods on the unlabeled values of the test series and reveal `is_anomaly` only
for evaluation.

| Method | Original/reference code | Upstream invocation and paper-specific setting |
|---|---|---|
| k-means | [scikit-learn `KMeans` source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_kmeans.py) | Install `scikit-learn`; make flattened multivariate windows of length 20 and stride 1, then use `KMeans(n_clusters=20).fit(train_windows)`. Score each window by distance to its assigned centroid and average overlapping window scores at every timestamp. |
| OCSVM | [scikit-learn `OneClassSVM` source](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_classes.py) | Install `scikit-learn`; call `OneClassSVM(...).fit(train)` and use `-decision_function(test)` as the anomaly score. Other parameters and thresholding follow the implementation defaults used in the paper. |
| MAD-GAN | [LiDan456/MAD-GANs](https://github.com/LiDan456/MAD-GANs) | After adding a TimeEval settings/data loader: `python RGAN.py --settings_file <dataset>` then `python AD.py --settings_file <dataset>`. |
| OmniAnomaly | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) | Add the converted dataset to `data_preprocess.py`, run `python data_preprocess.py <dataset>`, then `python main.py --dataset='<dataset>'`. |
| USAD | [manigalati/usad](https://github.com/manigalati/usad) | Put the converted arrays into the data-loading cells of `USAD.ipynb`, then execute the notebook top-to-bottom; the reusable model and training functions are in `usad.py`. |
| GDN | [d-ailin/GDN](https://github.com/d-ailin/GDN) | Create `data/<dataset>/{list.txt,train.csv,test.csv}` as described upstream, then run `bash run.sh <gpu_id> <dataset>` or `bash run.sh cpu <dataset>`. |
| AT | [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) | Add a canonical loader/config following `scripts/SMD.sh`, then run the corresponding script. Disable the repository's label-dependent adjustment and compute plain point-wise metrics for this comparison. |
| TranAD | [imperial-qore/TranAD](https://github.com/imperial-qore/TranAD) | Add the converted dataset in `preprocess.py`, run preprocessing, then `python3 main.py --model TranAD --dataset <dataset> --retrain`. |
| FuSAGNet | [seansihohan/FuSAGNet](https://github.com/seansihohan/FuSAGNet) | Create `data/<dataset>/{list.txt,train.csv,test.csv}` following its bundled SWaT example, select the dataset in its configuration, then run `python main.py`. |
| MTGFlow | [zqhang/MTGFLOW](https://github.com/zqhang/MTGFLOW) | Add the converted collection under `Dataset/input`, copy and edit the closest `runners/run_*.sh`, then execute it (for example, upstream uses `sh runners/run_WADI.sh`). |
| GCAD | [Tc99m/GCAD](https://github.com/Tc99m/GCAD) | Add a loader/config under `datasets`, edit the dataset and paths in `run.sh`, then run `bash run.sh`. |
| STAMP | [Matrix Profile Foundation reference implementation](https://github.com/matrix-profile-foundation/tsmp) | Compute one STAMP Matrix Profile per channel with subsequence length 100 and sum channel-wise profiles to form the multivariate score. A Python alternative is [`matrixprofile-ts`](https://github.com/target/matrixprofile-ts). |
| DAMP | [official DAMP code/documentation](https://sites.google.com/view/discord-aware-matrix-profile/documentation) | Run DAMP independently on every channel with subsequence length 100, then sum the channel-wise profiles. A runnable Python port is [`seansihohan/DAMP`](https://github.com/seansihohan/DAMP): `python damp.py --enable_output`. |

The paper's STAMP and DAMP numbers are from this straightforward channel-wise
multivariate extension. They should not be interpreted as results from a native
multivariate Matrix Profile detector. For AT-LSH, k-means, STAMP, and DAMP, the
decision threshold is the 99th percentile of training anomaly scores. No point
adjustment is used anywhere in Table 6 or Appendix A.
