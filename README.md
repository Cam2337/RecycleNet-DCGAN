# RecycleNet-DCGAN

<p align="center">
    <img src="https://user-images.githubusercontent.com/16138987/85941258-b35dc300-b8d6-11ea-8d7c-6c37955a1ee2.png"  style="width:400px;"/>
</p>

## Overview

This is the our team's final project codebase for Stanford's Spring 2020 offering of [CS230: Deep Learning](http://cs230.stanford.edu/). This project seeks to demonstrate the applicability of using machine-synthesized training data created by a deep convolutional generative adversarial network (DCGAN) to augment the capabilities of existing deep convolutional neural network (CNN) classifiers. Specifically, our project sought to use synthetic training data to augment the capability of an existing CNN recyclable waste classifier: [trashet](https://github.com/garythung/trashnet).

Training for approxmiately 9000 epochs on an AWS p2x.large EC2 instance, we were able to improve classification of one of the worst performing `trashnet` classes (metal) by nearly ~16% using synthetic data, while lowering classification accuracy by only several percentage points in the majority of other classification classes (with the largest reductions being in classes whose appareance shows strong resemblance to the metal training data).

Below is a snapshot of our results when training `trashnet` with an augmented dataset (original data plus the addition of synthetic metal images from RecyleNet-DCGAN) compared against a baseline run of `trashnet`. Be sure to check out our paper for a more in-depth look at our research and results.

<table>
<tr><th>Baseline Trashnet</th><th>Trashnet trained with augmented dataset</th>
<tr><td>

|   |G|Pa|C|Pl|M|T|Accuracy|
|---|---|---|---|---|---|---|---|
|G|50|6|4|14|3|5|60.976%|
|Pa|9|85|2|5|6|1|78.704%|
|C|1|3|56|5|3|2|80.000%|
|Pl|9|7|5|49|4|0|66.216%|
|M|9|7|6|5|37|4|54.412%|
|T|10|1|1|3|2|12|41.379%|

</td><td>

|   |G|Pa|C|Pl|M|T|Accuracy|
|---|---|---|---|---|---|---|---|
|G|48|2|2|12|15|3|58.537%|
|Pa|8|73|2|5|19|1|67.593%|
|C|3|1|54|3|8|1|77.143%|
|Pl|13|3|5|42|9|2|56.757%|
|M|11|5|2|0|48|2|70.588%|
|T|12|0|1|2|4|10|34.483%|


</td></tr></table>

## Dataset
This repository relies on a [forked version](https://github.com/Cam2337/trashnet) of the `trashnet` project for its underlying dataset. The data provided is the exact same data as what was provided by `trashnet`, with the exception that the contents in `dataset-resized.tar.gz` (avilable at the aformentioned `trashnet` fork) have been resized to 64x64 pixels for compatibility with the default generated images of `RecycleNet-DCGAN`. All credit for image collection goes to the creators of the [original trashnet project](https://github.com/garythung/trashnet). We ask that if you leverage this training data, please give a citation to the original `trashnet` repository.

The distribution of the raw data is as follows:
* 501 glass
* 594 paper
* 403 cardboard
* 482 plastic
* 410 metal
* 137 trash

The size of the original dataset exists on Google Drive. If you need access to the original dataset, **please give a citation to the [original trashnet project](https://github.com/garythung/trashnet)**. The dataset can be downloaded [here](http://drive.google.com/drive/folders/0B3P9oO5A3RvSUW9qTG11Ul83TEE).

More information about the underlying training data can be found in the `trashnet` submodule.

## Getting Started
### Linux - Ubuntu/Debian
Run `./setup.sh` to install the necessary dependencies. Dependencies for the project are as follows:

1. Install preqrequisite dependencies including `cmake`, `CUDA`, `GCC-4.8`, `G++-4.8`, and `Python3.6`
2. Install the [Torch](http://torch.ch/docs/getting-started.html) project, and setup the necessary Lua rocks for running trashnet
3. Setup the project's Python virtual environment in the project root directory, and install necessary Python3 packages

## Usage
### Step 1: Synthesize data using RecycleNet-DCGAN

To begin, ensure that your repository has its submodules initialized and ready. To work successfully, it will need to have the [trashnet fork](https://github.com/Cam2337/trashnet) cloned to the proper SHA.

From the root directory, you will find that the `src/` subdirectory contains all of the Python source code that you'll need. In addition to the Python source code, the `data/` directory contains symbolic links to the underlying `trashnet` submodule raw data used for training the `RecycleNet-DCGAN`<sup>[1](#f1)</sup>.

After performing initial setup in "Getting Started", you can begin training. Ensure that you are active in a local Python-3 virtual environment (venv). One should have already been created and setup for you after executing `setup.sh`:
```
source venv/bin/activate
```

From the root directory, you can begin training the `RecycleNet-DCGAN` as follows:
```
./train.py data/raw_data/metal-root/ 'metal' --batch-size 500 --image-size 64 --learning-rate 0.0002 --num-epochs 9000 2>&1 | tee results/metal_bs500_e9000.log
```

To learn what avialable hyperparameters are accepted by `train.py`, just run:
```
./train.py
```

Manual thresholds were set based on empircal evidence for when, after sufficient training, it was desireable to save synthetic images. As of the time of this writing, these thresholds are:
```
IMG_SAVE_COEF = 0.98
GAN_ERROR_THRESHOLD = 0.98
...
NUM_FAKES = 500
```

In other words, after `IMG_SAVE_COEF` of the epochs have been completed, iterations afterwards will be evaluated against a loss threshold (`GAN_ERROR_THRESHOLD`) and, should they be below said threshold, `NUM_FAKES` images will be generated and dumped to `results/figures/`.

### Step 2: Index the Synthesized Data

Now that you have properly synthesized recyclable waste data we need to tell the `trashnet` classifier about it! Within `src/`, a Python module  (`index_synthesized_images.py`) exists to do just that. The script can be invoked as follows:
```
./index_synthesized_images.py --in-dataroot results/figures/ \
--out-dataroot src/data/raw_data/metal-root/ \
--train-list submodules/trashnet/data/one-indexed-files-notrash_train.txt
```

In the above example, we're telling `index_synthesized_images.py` to place all of our synthetic metal images into the `src/data/dataset-resized/metal/` directory of the `trashnet` submodule. We're placing it there via the symlinks in `src/data/raw_data`<sup>[1](#f1)</sup>. Of note, the `--train-list` argument specifies the indexed list of images that `trashnet` should use for its training purposes. By adding references to our synthetic images here, this will instruct `trashnet` to use them during training, but **not** for validation or test.

### Step 3: Train/Validate/Test the `trashnet` model

We now have our synthetic images trained and indexed in the `trashnet` submodule. All we need to do now is to train `trashnet` itself. All of the logic necessary is in the `submodules/trashnet/train.lua` module. An example invocation of `train.lua` is as follows:
```
CUDA_LAUNCH_BLOCKING=1 th train.lua -dataFolder data/dataset-resized 2>&1 | tee trashnet_synth.log
```

Note that the above code excerpt was an example of invoking `train.lua` from the root of the `trashnet` submodule. The default run of [trashnet](https://github.com/Cam2337/trashnet) will perform training for 100 epochs, and assumes an input image size of 64x64. You can see more about the available hyperparameters and `trashnet` configuration from within the `trashnet` project itself.

### Step 4: View the results

At the end of training and validation, `train.lua` also calls into `test.lua`. All results are printed to `STDOUT`, along with a confusion matrix of the test performance.

## Contributing

Contribution instructions follow from the [trashnet project](https://github.com/garythung/trashnet):

1. Fork the repository
2. Create a feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push the branch: `git push -u origin my-new-feature`
5. Submit a pull request

## Acknowledgements

We would like to thank the creators of the `trashnet` project: [Gary Thung](https://github.com/garythung), and [Mindy Yang](https://github.com/yangmindy4), for their efforts to leverage AI for positive change. We would also like to thank Dr. Andrew Ng, the entire [CS230](http://cs230.stanford.edu/) TA staff, and give a personal thanks to our project mentor [Shahab Mousavi](https://github.com/ssmousav) for all of the support and guidance throughout the project.

<a id="f1">1.</a> The `RecyleNet-DCGAN` must train on **one** waste category at a time, so that a suitable training label can be assigned to the resulting synthetic images. To specify a single-class folder, the PyTorch [ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) class expects a hierarchy. Nesting references to the underlying raw data is a convenience mechanism to aid the user during training. See more at: [PyTorch#669]( https://github.com/pytorch/vision/issues/669).