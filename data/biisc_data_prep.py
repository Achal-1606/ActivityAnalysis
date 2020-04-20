import os
from pathlib import Path


BASE_DIR = Path(__file__).parent
SOURCE_DIR = BASE_DIR / "biisc"
DATA_METADATA_DIR = BASE_DIR / "biiscTrainTestlist"

CLASS_AVAILABLE = [
    'CALL',
    'COUGH',
    'DRINK',
    'SCRATCH',
    'SNEEZE',
    'STRETCH',
    'WAVE',
    'WIPE'
]

TEST_VERSION = '01'
TEST_SUBJS = ['S002', 'S003', 'S004', 'S005', 'S006']

if not DATA_METADATA_DIR.exists():
    print("Metadata directory not available...Creating ",
          DATA_METADATA_DIR.name)
    os.mkdir(str(DATA_METADATA_DIR))

print("Writing Class Indicator file...")
with open(str(DATA_METADATA_DIR / "classInd.txt"), 'w') as f:
    f.writelines(
        ['{} {}\n'.format(num + 1, action)
         for num, action in enumerate(CLASS_AVAILABLE)])

print("Creating Train & Test list from the directory")
train_list, test_list = [], []
action_list = [x[:4] for x in CLASS_AVAILABLE]
for video in (SOURCE_DIR / "videos").iterdir():
    fname = video.name
    subject = fname.split('_')[0]
    action = fname.split('_')[2]
    class_name = CLASS_AVAILABLE[action_list.index(action)]
    if subject in TEST_SUBJS:
        test_str = "{}/{}\n".format(class_name, video.name)
        test_list.append(test_str)
    else:
        train_str = "{}/{} {}\n".format(
            class_name, fname, action_list.index(action) + 1)
        train_list.append(train_str)

print("train Samples - {} | Test Samples - {}".format(len(train_list), len(test_list)))
print("-----------Sample Train list ------------- \n", train_list[23:29])
print("-----------Sample Test list ------------- \n", test_list[23:29])

print("Writing train file Version - ", TEST_VERSION)
with open(str(DATA_METADATA_DIR / "trainlist{}.txt".format(TEST_VERSION)), 'w') as f:
    f.writelines(train_list)

print("Writing test file Version - ", TEST_VERSION)
with open(str(DATA_METADATA_DIR / "testlist{}.txt".format(TEST_VERSION)), 'w') as f:
    f.writelines(test_list)
