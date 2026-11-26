# CIFAR-10 as used in this assignment

CIFAR-10 has 60,000 colour images. Each image is 32 by 32 pixels with three channels (red, green, blue). There are 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck. The official division is 50,000 training images and 10,000 test images, with 5,000 and 1,000 images in each class.

This project does not train on all 50,000. Seed 26003 (the digits of roll m26csa003, in order) draws one stratified split of the official training set: 45,000 train images and 5,000 validation images, which is 4,500 and 500 in every class. The indices are stored in `splits/split_seed26003.npz`. Train and validation do not overlap, and together they are exactly the official training set. The official test set is a separate array of 10,000 images and is not part of that file.

The validation images are reserved for choosing models, architectures and hyperparameters. The test images are scored only after those choices are fixed.

Per-channel mean and population standard deviation are computed from the 45,000 training images only, after scaling pixels from 0–255 into [0, 1]:

- mean (R, G, B) = [0.491432, 0.482107, 0.446561]
- std (R, G, B) = [0.246945, 0.243417, 0.261535]

These constants are saved in `splits/norm_stats_seed26003.npz`. Validation and test pixels were not included. The often-quoted CIFAR-10 moments are estimated on all 50,000 official training images, so they are close to these numbers but they are not the ones used here.

Two properties of the data matter for the models in later questions. The images are small: a 32 by 32 picture loses fine structure such as text, thin legs and the difference between similar outlines (cat versus dog, automobile versus truck). Within one class the pictures still vary a lot in pose, colour, background and where the object sits. A classifier that cannot reuse a local pattern at many positions has to relearn that pattern separately in every part of the image.

The class-count figure uses test labels only to show that the official test set is balanced. No test pixel was reduced into a saved statistic, and no model was selected in this step.
