# Report notes

Praveen, roll m24ma2004, seed 242004.

Q1–Q4 have measured results. There is still no git commit, so page 1 has no commit hash yet.

Sample standard deviation is the one with divisor n−1. Percentages below are the logged fractions times 100.

## Dataset section

| Item | Where it is produced |
| --- | --- |
| Split sizes, class counts, overlap, index hash | `scripts/prepare_data.py` writes `splits/verification_seed26003.json` |
| Train-only mean and std | `src/common/data.py` `channel_mean_std`, saved as `splits/norm_stats_seed26003.json` |
| Class-count table | `results/phase1/class_counts.csv` |
| Per-class train pixel moments | `results/phase1/train_per_class_pixel_stats.csv` |
| Sample grid | `figures/class_samples.png` and `.pdf` |
| Class distribution bars | `figures/class_distribution.png` and `.pdf` |
| Mean image per class | `figures/mean_images.png` and `.pdf` |
| Pixel histograms | `figures/pixel_statistics.png` and `.pdf` |
| Written dataset description | `figures/dataset_description.md` |

Regenerate the figures with `PYTHONNOUSERSITE=1 .venv/bin/python scripts/prepare_data.py`. They are drawn from the saved split, not from a new random draw.

Train index SHA-256 `97af65b491226b292df93e2a67fa52812257f3db3c0f045799d4eb675c70f04a`. Validation index SHA-256 `51c33528721993c8a467510f7cb42cbc6a612d0a06d7677daa6feaadbf1df23c`. Sizes 45,000 / 5,000 / 10,000. Per class 4,500 / 500 / 1,000. Overlap 0.

## Q1a

The comparison was locked in `configs/q1_fairness.json` at 2026-11-10T13:00:14+05:30, sha256 `83592a28756f0a14e26ba972e6b82548e1af5ad2862d5af39b06ca7cba9ac09c`. Do not edit that file to match these numbers.

| Item | File |
| --- | --- |
| Learning-rate probe scores | `results/q1/chosen_lr.json` |
| Per-seed histories | `results/q1/final/*.json` and `.csv` |
| Test accuracies of the selected checkpoints | `results/q1/test_results.json` |
| Mean ± sample std table | `results/q1/q1a_model_table.csv` and `.tex` |
| Curves | `figures/q1/q1a_train_accuracy`, `q1a_val_accuracy`, `q1a_train_loss`, `q1a_val_loss`, `q1a_lr_probe`, `q1a_test_accuracy` (png and pdf) |

Probe scores (mean of the two models' best validation accuracies, 8 epochs, seed 26003): 0.0003 → 0.5341, 0.001 → 0.6022, 0.003 → 0.6263. Chosen learning rate 0.003. The matched MLP's own probe scores were 0.5072, 0.5316, and 0.5304, so 0.001 and 0.003 were almost tied for the MLP. The CNN moved from 0.561 to 0.7222 and dominated the average.

Final models, 30 epochs, seeds 26003, 26004, 26005, official test after validation selection:

| Model | Params | Val accuracy | Test accuracy |
| --- | --- | --- | --- |
| Matched MLP | 201,482 | 0.515000 ± 0.003105 | 0.504367 ± 0.001801 |
| Matched CNN | 197,194 | 0.741933 ± 0.031390 | 0.738767 ± 0.030222 |
| Wide MLP | 1,707,274 | 0.532733 ± 0.005085 | 0.526667 ± 0.002627 |
| Regularised MLP | 201,482 | 0.513667 ± 0.003035 | 0.504600 ± 0.003851 |

The CNN is about 23 test points ahead of the matched MLP. About 8.5× more parameters add about 2 test points. Dropout 0.3 and weight decay 0.001 do not raise test accuracy. The weakest CNN seed is still well above the best wide-MLP seed. The wide MLP's training accuracy is high and its test loss is high, so the extra parameters are used to overfit, not to close the gap.

## Q1b

Same wide MLP as Q1a, first introduced there. This run turns weight decay off, which the fairness file says is not part of the fair Q1a comparison. Seed 26003, 40 epochs, learning rate 0.003. Corruption masks: `splits/q1b_noise/fraction_{0,0.2,0.5,1}_seed26003.npz`.

| Item | File |
| --- | --- |
| Epoch histories | `results/q1/q1b/mlp_wide_noise{0,0.2,0.5,1}_seed26003.csv` |
| Peak and final line | `results/q1/q1b_summary.csv` |
| Epochs quoted below | `results/q1/q1b_milestones.csv` |
| Plot | `figures/q1/q1b_memorisation.png` and `.pdf` |

Chance on 10 classes is 0.10. A corrupted label is never the true class.

| Noise | What happened |
| --- | --- |
| 0% | Validation peaks at 0.5336 on epoch 33, with clean-label training accuracy 0.9017. By epoch 40 clean accuracy is 0.9180 and validation is 0.5328. |
| 20% | Epoch 1: validation 0.3930, noise fit 0.0702. Epoch 5: validation 0.4582, noise fit 0.0808. Peak validation 0.4634 on epoch 19, noise fit already 0.2282. Epoch 40: validation 0.4392, noise fit 0.3947, accuracy on the true labels of corrupted rows 0.1369. |
| 50% | Epoch 1: validation 0.2948, noise fit 0.0833. Peak validation 0.3184 on epoch 6, noise fit 0.0902. Epoch 40: validation 0.2816, noise fit 0.2848, true-label accuracy on corrupted rows 0.1392. |
| 100% | Training loss stays at about 2.3024–2.3027, which is ln(10). Validation stays at 0.100 and the fit to the random labels stays at about 0.103 for all 40 epochs. |

Reading against the sentence written before training: at 20% and at 50%, validation does its rising while the fit to the random labels is still near chance, and that fit keeps rising after validation has peaked. That is the condition under which the plan said useful structure appeared before memorisation. The 20% rise is about six points. The 50% rise is about two points, so the separation is real there but the useful gain is small. The two curves do not rise together through the useful part.

The 100% run does not support the guess that the random-label fit would climb. This budget did not memorise a fully random training set. The 20% and 50% runs are the measurements that show partial memorisation.

## Q2

Plan `configs/q2_plan.json`, written 2026-11-10T17:48:52+05:30, before the permuted and grayscale training. Q2a reuses the Q1a matched MLP (201,482) and matched CNN (197,194). Same learning rate 0.003, 30 epochs, seeds 26003–26005.

| Item | File |
| --- | --- |
| Per-seed test accuracy, confidence, confusion | `results/q2/test_results.json` |
| Mean ± sample std | `results/q2/q2_summary.csv` and `.tex` |
| Plots | `figures/q2/q2_test_accuracy`, `q2_confidence`, `q2_confusion_cnn_seed26003` |

Test accuracy, mean ± sample std:

| Model | Clean | Pixels permuted | Grayscale |
| --- | --- | --- | --- |
| Matched MLP | 0.504367 ± 0.001801 | 0.501133 ± 0.002854 | 0.418633 ± 0.004901 |
| Matched CNN | 0.738767 ± 0.030222 | 0.485567 ± 0.028458 | 0.611967 ± 0.122399 |

The permutation leaves the MLP at 50.1% and drops the CNN from 73.9% to 48.6%, onto the MLP. That is the change the plan said would support missing spatial weight sharing.

Grayscale keeps a gap on the mean (61.2% versus 41.9%). The CNN seeds are 0.7146, 0.6448 and 0.4765. Two seeds stay above 60% and well above the MLP. Seed 26005 has validation accuracy 0.4898 and test accuracy 0.4765, so the “above 60%” sentence is not true of every seed. Both models lose accuracy when colour is removed, so colour is used. It does not account for the clean-image gap on the two stable seeds.

Matched CNN mean max-softmax, correct versus incorrect: clean 0.862 versus 0.648, permuted 0.599 versus 0.440, grayscale 0.717 versus 0.515. The fraction of errors with confidence at least 0.9 falls from 0.148 on clean images to 0.008 after the permutation.

## Q3

Plan `configs/q3_plan.json`, same timestamp. Learning rate 0.003 from the middle-bottleneck probe (`results/q3/chosen_lr.json`). Validation MSE: bottleneck 32 → 0.011525 (267,843 parameters), 128 → 0.005571 (661,155), 512 → 0.004789 (2,234,403). All three checkpoints are epoch 20, the last epoch of the budget. Selection file `results/q3/q3a_selection.json` chooses 512. Test MSE of that model only, written afterwards: 0.004834, PSNR 23.16 (`results/q3/q3a_test_mse.json`).

Frozen classifier: Q1a matched CNN, seed 26003, the highest validation accuracy. Test classification (`results/q3/q3b_classification.csv`):

| Input | Accuracy | Pixel MSE | PSNR |
| --- | --- | --- | --- |
| Clean | 0.7612 | 0 |  |
| Gaussian noise, sigma 0.25 | 0.1670 | 0.048385 | 13.15 |
| Clean autoencoder of the noisy image | 0.3345 | 0.009735 | 20.12 |
| Denoising autoencoder of the noisy image | 0.3296 | 0.007573 | 21.21 |

The denoiser has the better pixels. Classification of the two reconstructions is about 33%, against 16.7% for the raw noisy image and 76.1% for the clean image. Denoising does not restore the accuracy the classifier had on clean images.

Linear probes, C chosen on validation (`results/q3/q3c_representation.csv`). Test MSE and test probe accuracy: 32 → 0.011603 and 0.3640; 128 → 0.005627 and 0.4438; 512 → 0.004834 and 0.5453. Wider codes are better on both. Most of the MSE drop is from 32 to 128. The probe keeps rising from 128 to 512, where the MSE drop is small. The denoiser’s clean-image code, same width 512, has probe test accuracy 0.5392. Some logistic fits hit the 400-iteration limit, so an individual C can be slightly short of its optimum. The three probe accuracies are separated by about eight points.

## Q4

Plan `configs/q4_plan.json`, written before any fit. Representation: Q1a matched CNN, seed 26003, 160-d pooled features. Subset: 200 per class, 2,000 rows, `splits/q4_svm_subset_seed26003.npy`.

Full sweep `results/q4/svm_sweep.csv`. Plots `figures/q4/q4_linear.png`, `q4_poly.png`, `q4_rbf.png`.

Selected on validation: polynomial kernel, degree 3, C = 1. Validation accuracy 0.734, training accuracy 0.999, support-vector fraction 0.6855. One test evaluation: 0.7235 (`results/q4/svm_test.json`).

Small C underfits. RBF at C = 0.01 has training accuracy 0.693, validation 0.609, and every training point is a support vector. Linear at C = 0.01 is the best linear setting (validation 0.730). Raising linear C to 1 and above drives training accuracy to 0.998–1.000 and validation down to about 0.709. Polynomial and RBF peak at C = 1 and C = 10, then C = 10 and C = 100 are identical.

Hard pairs by test confusion (`results/q4/hard_pairs.json`, figure `figures/q4/q4_hard_pairs.png`): cat/dog 358 confusions out of 2,000 pair images (rate 0.179, 222 support vectors); bird/deer 178 (0.089, 171 support vectors); airplane/ship 146 (0.073, 139 support vectors). The cat/dog test points with the smallest absolute decision scores are misclassified, with scores about 0.001 and −0.002.

Clustering: KMeans, k chosen by validation NMI. k = 10, validation NMI 0.362. Test ARI 0.228, test NMI 0.361, majority-vote accuracy 0.460 against the SVM’s 0.724. The disagreement class under the written rule is bird: SVM recall 0.615, cluster majority recall 0.000, because no training cluster had bird as its majority label. Contingency and recall bars: `figures/q4/q4_contingency.png`, `q4_recall_comparison.png`.

## Reused pieces

| Piece | First introduced |
| --- | --- |
| Stratified split and loader (`src/common/data.py`) | shared setup, before Q1 |
| Seed / device helpers (`src/common/utils.py`) | shared setup, before Q1 |
| MLP and ConvNet (`src/common/models.py`) | Q1a |
| Wide MLP used in Q1b | Q1a (`mlp_wide`); Q1b only changes weight decay to 0 |
| Matched MLP and matched CNN in Q2 | Q1a |
| Frozen CNN classifier in Q3b and the Q4 features | Q1a, checkpoint seed 26003 |
| Clean autoencoder used as the Q3b baseline | Q3a, bottleneck 512 |
