# Activity Recognition in Video

There is a detailed presentation in `docs` folder which detailed out in steps our full approaches.

Some ideas related to Activity recognition methods are:

1. Classify one frame at a time with a ConvNet
1. Extract features from each frame with a ConvNet, passing the sequence to an RNN, in a separate network
1. Use a time-dstirbuted ConvNet, passing the features to an RNN, much like #2 but all in one network (this is the `lrcn` network in the code).
1. Extract features from each frame with a ConvNet and pass the sequence to an MLP
1. Use a 3D convolutional network (has two versions of 3d conv to choose from)

We are currently trying to do the recognition using the Method #2 & #4 from the methods mentioned above. However we are still trying to do work on evaluating all of the above mentioned methods.

## Requirements

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. 

To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files. If `ffmpeg` isn't in your system path (ie. `which ffmpeg` doesn't return its path, or you're on an OS other than *nix), you'll need to update the path to `ffmpeg` in `data/2_extract_files.py`.

## Getting the data

First, download the dataset from UCF into the `data` folder:

`cd data && wget http://crcv.ucf.edu/data/UCF101/UCF101.rar`

Then extract it with `unrar e UCF101.rar`.

Next, create folders (still in the data folder) with `mkdir train && mkdir test && mkdir sequences && mkdir checkpoints`.

Now you can run the scripts in the data folder to move the videos to the appropriate place, extract their frames and make the CSV file the rest of the code references. You need to run these in order. Example:

`python 1_move_files.py`

`python 2_extract_files.py`

## Extracting features

Before you can run the `lstm` and `mlp`, you need to extract features from the images with the CNN. This is done by running `extract_features.py`. On my Dell with a GeFore 960m GPU, this takes about 8 hours. If you want to limit to just the first N classes, you can set that option in the file.

## Training models

The CNN-only method (method #1 above) is run from `train_cnn.py`.

The rest of the models are run from `train.py`. There are configuration options you can set in that file to choose which model you want to run. If you are trying to run using Tensorflow 2, set `load_to_memory = True` in train.py.

The models are all defined in `models.py`. Reference that file to see which models you are able to run in `train.py`.

Training logs are saved to CSV and also to TensorBoard files. To see progress while training, run `tensorboard --logdir=data/logs` from the project root folder. The models will get stored in the `data/checkpoints` directory

## To run a sample video

1. `mkdir demo_dir`
1. `cd demo_dir && mkdir input_vid && mkdir video_frames && mkdir video_features && output_vid`
1. Place your video for classification in `demo_dir/input_vid`
1. `python train.py`
