# Explainer

Praveen, roll m26csa003. This is the viva walkthrough. Numbers come from the files listed in `REPORT_NOTES.md`.

## Q1. Why a plain MLP is weak on CIFAR-10

### Idea

A convolution reuses one small filter at every position. An MLP has a separate weight for every pixel, so a pattern that slides a few pixels is a new pattern. The question is whether that missing reuse is why the MLP loses, or whether the MLP is just too small or just overfits.

### Design

Written down in `configs/q1_fairness.json` before any training.

- Matched MLP: 64 then 64 hidden units, 201,482 parameters, dropout off, weight decay 1e-4.
- Matched CNN: three 3×3 convolutions (64, 96, 160), max-pool, global average pool, 197,194 parameters. No batch norm.
- Wide MLP: 512 then 256, 1,707,274 parameters. Same optimiser, epochs, learning rate, and images. This is the capacity control.
- Regularised MLP: same widths as the matched MLP, dropout 0.3, weight decay 1e-3. This is the overfitting control. No augmentation on any of them.

One learning rate for every model. An 8-epoch probe on the matched MLP and the matched CNN scores each candidate by the average of their best validation accuracies. The higher score wins. The wide and regularised models do not get their own search. Thirty epochs, three seeds (26003, 26004, 26005). The checkpoint is the earliest epoch with the highest validation accuracy. The test set is scored after that choice.

### What I expected

A gap of about 10 points or more, with the MLP somewhere in 45–60% and the CNN in 60–75% if there is no augmentation. The wide MLP should beat the narrow one and still lose to the CNN if weight sharing is the real reason. Regularisation should not close a large gap.

### What happened

The probe picked learning rate 0.003. Test accuracy, mean ± sample std over the three seeds:

- Matched MLP 50.4% ± 0.2
- Matched CNN 73.9% ± 3.0
- Wide MLP 52.7% ± 0.3
- Regularised MLP 50.5% ± 0.4

The gap is about 23 points, larger than the guess, and in the same direction. Extra parameters add about 2 points. Regularisation does not help. One CNN seed is weaker than the other two (the ±3 point spread); it is still far above every MLP seed.

The probe average is driven by the CNN. The MLP's own 8-epoch scores at 0.001 and 0.003 were almost the same, so the shared learning rate is not an obvious handicap for the MLP.

### Likely questions

- Why no batch norm and no augmentation? Both would help the CNN for reasons other than weight sharing. Leaving them out keeps the comparison on the architecture.
- Is 8.5× parameters "enough capacity"? It is the control we locked. It overfits (high train accuracy, high test loss) and still does not approach the CNN. A still-wider net was not part of the plan.
- Why is the CNN's seed spread large? It is reported, not hidden. The conclusion uses the mean, and the weakest seed still wins.

## Q1b. Structure before memorisation

### Idea

Give the wide MLP labels that should not generalise, and watch four curves every epoch: accuracy on the clean training rows, fit to the random labels, accuracy on the true class of the corrupted rows, and clean validation accuracy.

### Design

Same widths as the Q1a wide MLP. Weight decay is 0 so it does not fight the random labels. This run is not part of the fair Q1a table. Fractions 0, 20%, 50%, 100%, stratified inside each class, replacement drawn from the other nine classes. Validation labels stay clean. Forty epochs, seed 26003, the same learning rate 0.003.

The rule written beforehand: say that useful structure came before memorisation only if, at 20% or 50%, validation rises while the fit to the random labels is still near chance (about 10%), and that fit keeps rising after validation has peaked. If the two rise together, do not separate them.

### What happened

At 20%, validation goes from 39.3% to 45.8% in the first five epochs while the random-label fit stays at 7–8%. The peak is 46.3% at epoch 19, by which time the noise fit is already 22.8%. After that the noise fit climbs to 39.5% and validation slips to 43.9%. The true-label accuracy on corrupted rows falls from 38.6% to 13.7%: the net starts calling those images by the random label.

At 50%, validation peaks at 31.8% on epoch 6 while the noise fit is still 9.0%. By epoch 40 the noise fit is 28.5% and validation is 28.2%. The useful rise is only about two points.

At 100%, nothing moves. Training loss stays at ln(10) and both validation and the random-label fit stay at chance. The guess that a fully random training set would be fit in 40 epochs was wrong for this learning rate. The partial-noise runs are the ones that show memorisation.

So the pre-written condition holds at 20% and, more weakly, at 50%. I will not say it for the fully random set, because that set was not fit.

### Likely questions

- Why is weight decay off here but on in Q1a? Q1a is a fair comparison. Q1b asks whether the net can fit bad labels. Weight decay works against that, so it is off, and the two tables are not mixed.
- Did the early validation rise at 20% already include some memorisation? The noise fit is still below 10% through epoch 7. Most of the validation gain is in that window. The last half point up to the peak does overlap the start of the noise fit, which is why the claim is about the early window, not about epoch 19 being "clean".
- One seed only. A second seed could move the 50% peak. The 20% pattern is large enough that a small seed shift would not remove it, but that is a judgement, not a measured standard deviation.

## Q2. What the CNN is using

### Idea

Q2a reuses the Q1 matched pair and the reason already measured there: the CNN can reuse a local pattern and the MLP cannot. Q2b deletes spatial neighbourhoods and keeps the pixel values. Q2c deletes colour and keeps the neighbourhoods. The same transform is used on train, validation and test.

### What I expected

After one fixed pixel permutation, the MLP should stay within about 3 points of 50.4%, and the CNN should fall by much more than 10 points toward the MLP. On grayscale, if shape is the reason, the CNN should stay more than 10 points above the MLP and above 60%.

### What happened

Permutation: MLP test accuracy 50.1% ± 0.3, CNN 48.6% ± 2.8. The CNN’s clean accuracy was 73.9%. The spatial explanation holds.

Grayscale: MLP 41.9% ± 0.5, CNN 61.2% ± 12.2. The CNN seeds are 71.5%, 64.5% and 47.7%. Two seeds match the shape story. Seed 26005 does not: its validation accuracy is 49.0%. Colour hurts both models, and it is not a stable explanation of the whole CNN lead.

The CNN is more confident when it is right than when it is wrong in every condition. After the permutation its confidence falls, and errors with confidence above 0.9 almost disappear (14.8% of errors on clean images, 0.8% after the permutation).

### Likely questions

- Why is the grayscale mean misleading? Because one seed failed. Quote the three seeds, not only the mean.
- Could the permutation have changed the MLP’s function class? No. The same reordering of positions is a reordering of the first-layer weights. The measured accuracies match that.

## Q3. Bottlenecks and denoising

### Idea

Three autoencoders differ only in the width of the code: 32, 128, 512. Pick the lowest validation MSE. Then train a denoiser of that width on Gaussian noise with standard deviation 0.25, and ask whether the reconstruction gives a frozen Q1 CNN its class decision back. Separately, a logistic regression reads each frozen code.

### What happened

Validation MSE falls with width: 0.0115, 0.0056, 0.0048. The rule picks 512. Its test MSE is 0.00483 (PSNR 23.2). Every model was still at its best on the last epoch, so a longer budget might lower all three. The order is already clear.

The frozen CNN scores 76.1% on clean test images and 16.7% on the noisy ones. Reconstructing with the clean autoencoder raises that to 33.5%. The denoiser has a better PSNR (21.2 against 20.1) and a classification accuracy of 33.0%. Better pixels did not recover the classifier. Both reconstructions help relative to raw noise and leave most of the gap to the clean image.

Probe test accuracy is 36.4%, 44.4%, 54.5% as the code gets wider. Reconstruction and class information move the same way. The MSE gain is concentrated between 32 and 128. The probe gain continues between 128 and 512. Some fits stopped at 400 iterations, so a single C can be slightly short of the best logistic fit. The gaps between bottlenecks are larger than that.

## Q4. SVM and clusters on the frozen CNN code

### Idea

The 160-d vector from the Q1 CNN, seed 26003, is frozen. Every SVM sees the same 2,000 stratified training rows, chosen before any C was fit. Five values of C from 0.01 to 100, and linear, polynomial and RBF kernels. Validation picks one setting. That setting is tested once.

### What happened

Polynomial degree 3 with C = 1 wins validation at 73.4%, with training accuracy 99.9% and 68.6% of the training rows as support vectors. The test accuracy is 72.4%.

RBF with C = 0.01 underfits: 69.3% train, 60.9% validation, and every training point is a support vector. Linear with a large C overfits: training accuracy reaches 100% and validation falls from 73.0% at C = 0.01 to about 71%. Polynomial and RBF are flat from C = 10 to C = 100.

The hardest test pairs are cat/dog (358 confusions, 17.9% of that pair), bird/deer (178, 8.9%) and airplane/ship (146, 7.3%). The cat/dog test images closest to the boundary are errors, with decision scores near zero.

KMeans with k = 10 was chosen by validation NMI. Test ARI is 0.228, test NMI is 0.361, and majority-vote accuracy is 46.0%, against the SVM’s 72.4%. Bird is the disagreement: the SVM recalls it at 61.5%, and no training cluster has bird as its majority label, so the cluster map’s bird recall is 0.

### Likely questions

- Why 2,000 points and not 45,000? A kernel SVM on the full training split is the expensive case the assignment allows a subset for. The subset is stratified, saved, and shared by every kernel and every C.
- Why is bird recall zero for clustering if bird images exist? The map is the majority training label of each cluster. Bird images are mixed into clusters whose majority is another class, so the map never outputs bird.
