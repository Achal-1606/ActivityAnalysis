"""
Given a video path and a saved model (checkpoint), produce classification
predictions.

Note that if using a model that requires features to be extracted, those
features must be extracted first.

Note also that this is a rushed demo script to help a few people who have
requested it and so is quite "rough". :)
"""
from keras.models import load_model
from data import DataSet
import numpy as np
from os.path import *
import os
import glob
from subprocess import call
from extractor import Extractor
import cv2
import pandas as pd


def predict(data_type, seq_length, saved_model, image_shape, video_name,
            class_limit):
    model = load_model(saved_model)

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(seq_length=seq_length, class_limit=class_limit)
    else:
        data = DataSet(seq_length=seq_length, image_shape=image_shape,
                       class_limit=class_limit)

    # Extract the sample from the data.
    sample = data.get_frames_by_filename(video_name, data_type)

    # Predict!
    prediction = model.predict(np.expand_dims(sample, axis=0))
    print(prediction)
    data.print_class_from_prediction(np.squeeze(prediction, axis=0))


def get_video_frames(video_name, base_directory):
    path = base_directory
    images = sorted(glob.glob(join(path, video_name + '*jpg')))
    return images


def get_image_text(label_prediction):
    max_label = sorted(label_prediction.items(), key=lambda x: x[1],
                       reverse=True)[0][0]
    out_text = ['Pred - {}'.format(max_label)]
    for key in label_prediction:
        out_text.append(key + ' : ' + '{0:.5f}'.format(label_prediction[key]))
    return out_text


def write_predicted_images(image, label_pred_dict):
    # out_dir, file_name = img_out_path
    img_text = get_image_text(label_pred_dict)
    img_array = []
    size = None
    for filename in image:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        start = 10
        for text in img_text:
            cv2.putText(img, text,
                        (10, start),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 255),
                        1)
            start += 10
        img_array.append(img)
    return img_array, size


def main():
    # model can be one of lstm, lrcn, mlp, conv_3d, c3d.
    model = 'lstm'
    # Must be a weights file.
    # saved_model = 'data/checkpoints/lstm-features.026-0.239.hdf5'
    saved_model = 'data/checkpoints/lstm-features.026-0.038.hdf5'
    # Sequence length must match the lengh used during training.
    seq_length = 40
    # Limit must match that used during training.
    class_limit = 4

    # Chose images or features and image shape based on network.
    if model in ['lstm', 'mlp']:
        data_type = 'features'
        image_shape = None
    else:
        raise ValueError("Invalid model. See train.py for options.")

    # Creating Directory Structure
    demo_dir = join(dirname(__file__), 'demo_dir')
    video_dir = join(demo_dir, 'input_vid')
    process_dir = join(demo_dir, 'video_frames')
    features_dir = join(demo_dir, 'video_features')
    output_dir = join(demo_dir, 'output_vid')
    data_csv_file = join(dirname(__file__), 'data', 'data_file.csv')
    # Classes for evaluation
    data_pd = pd.read_csv(data_csv_file, header=None,
                          names=['test_train', 'class', 'file', 'frames'])
    class_list = sorted(data_pd['class'].unique().tolist())

    # Demo file. Must already be extracted & features generated
    # (if model requires)
    # Do not include the extension.
    # Assumes it's in data/[train|test]/
    # It also must be part of the train/test data.
    # TODO Make this way more useful. It should take in the path to
    # an actual video file, extract frames, generate sequences, etc.
    # video_name = 'v_Archery_g04_c02'
    # video_name = 'v_ApplyLipstick_g01_c01'
    # video_name = 'v_WalkingWithDog_g01_c01'

    # Clear the processing dir
    if exists(process_dir):
        files = glob.glob(join(process_dir, '*'))
        for file in files:
            os.remove(file)
        print('Frames Directory {} cleared'.format(process_dir))
    else:
        os.mkdir(process_dir)

    # Clear the feature dir
    if exists(features_dir):
        files = glob.glob(join(features_dir, '*'))
        for file in files:
            os.remove(file)
        print('Features Directory {} cleared'.format(features_dir))
    else:
        os.mkdir(features_dir)

    model = load_model(saved_model)
    extractor_model = Extractor()

    # Process the input video
    vid_files = glob.glob(join(video_dir, '*'))
    for video in vid_files:
        # Extract the video into frames (images)
        src = video
        filename_no_ext = split(src)[-1].split('.')[0]
        dest = join(process_dir, filename_no_ext + '-%04d.jpg')
        call(["ffmpeg", "-i", src, dest])
        # predict(data_type, seq_length, saved_model, image_shape,
        #         split(video)[-1], class_limit)
        # np_feature_file = join(features_dir, filename_no_ext + )
        # if os.path.isfile(path + '.npy'):
        #     pbar.update(1)
        #     continue
        feature_path = join(features_dir, filename_no_ext +
                            '-all-frames-features.npy')
        # Get the frames for this video.
        frames = get_video_frames(filename_no_ext, process_dir)

        # Now downsample to just the ones we need.
        # frames = data.rescale_list(frames, seq_length)

        # Now loop through and extract features to build the sequence.
        sequence = []
        for image in frames:
            features = extractor_model.extract(image)
            sequence.append(features)

        # Save the sequence.
        # np.save(feature_path, sequence)
        size = None
        for i in range(0, len(sequence), seq_length):
            start, end = (i, i + seq_length)
            if end >= len(sequence):
                # Reject the frame sequence, as less than seq_length
                continue
            batch_seq = np.array(sequence[start: end])
            batch_frames = np.array(frames[start:end])
            prediction = model.predict(np.expand_dims(batch_seq, axis=0))
            label_predictions = {}
            for num, label in enumerate(class_list):
                label_predictions[label] = prediction[0][num]
            img_arr, size = write_predicted_images(batch_frames,
                                                   label_predictions)
            if i == 0:
                proc_image_arr = img_arr
            else:
                proc_image_arr = np.append(proc_image_arr, img_arr, axis=0)
        out = cv2.VideoWriter(
            join(output_dir, 'processed-' + filename_no_ext + '.avi'),
            cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

        for i in range(len(proc_image_arr)):
            out.write(proc_image_arr[i])
        out.release()


if __name__ == '__main__':
    main()
