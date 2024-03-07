import os
import yaml
import json
import shutil
import numpy as np
from tqdm import tqdm
import argparse
import cv2
# import sys
from warnings import warn


# print("sysexe:",sys.executable)
parser = argparse.ArgumentParser()
parser.add_argument("-source_dir", type=str, help="Source dir of the PVDN dataset",
                    default="/home/lukas/Development/datasets/PVDN/day")
parser.add_argument("-target_dir", type=str, help="Target dir of the new Yolo format "
                    "dataset.", default="/home/lukas/Development/datasets/PVDN_yolo/day")
parser.add_argument("-img_size", type=int, help="Final yolo image size (image will be square).", default=960)
args = parser.parse_args()

target_dir="F:\\OUTPUT"
target_dir = os.path.abspath(target_dir)
os.makedirs(target_dir, exist_ok=True)

source_dir = "F:\\S00017"
if not os.path.isdir(source_dir):
    raise NotADirectoryError(f"{source_dir} is not a directory. Please check.")
if not os.path.isdir(target_dir):
    raise NotADirectoryError(f"{target_dir} is not a directory. Please check.")

num_classes = 1
names = ['instance']

overview = {
    "train": os.path.join(target_dir, "train"),
    "val": os.path.join(target_dir, "val"),
    "test": os.path.join(target_dir, "test"),
    "nc": num_classes,
    "names": names
}

# create .yaml file required for yolo training
yaml_dir = os.path.join(target_dir, 'pvdn.yaml')
print(f"Creating yolo .yaml file at {yaml_dir}...")
with open(yaml_dir, "w") as f:
    yaml.dump(overview, f, default_flow_style=None)

# doing conversion for each split
splits = ("train", "test", "val")
for split in splits:

    # checking & setting up paths
    target_path = os.path.join(target_dir, split)
    source_path = os.path.join(source_dir, split)
    if not os.path.isdir(source_path):
        warn(f"{source_path} does not exist or is not a directory. Skipping the {split} split.")
        continue
    os.makedirs(target_path, exist_ok=True)

    print(f"Copying {split} images to {target_path}.")
    scenes_dir = os.path.join(source_path, "images")
    scenes = os.listdir(scenes_dir)
    for scene in tqdm(scenes, desc=f"Running through scenes of the {split} split"):
        images = os.listdir(os.path.join(scenes_dir, scene))
        for img in images:
            # resize image to be square (img_size x img_size)
            im = cv2.imread(os.path.join(scenes_dir, scene, img), 0)
            h_orig, w_orig = im.shape
            im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)

            # save image to new location
            cv2.imwrite(os.path.join(target_path, img), im)
            if not os.path.exists(os.path.join(target_path, img)):
                shutil.copy(os.path.join(scenes_dir, scene, img), target_path)

            # create annotation file
            annot_file = img.split(".")[0] + ".json"
            with open(os.path.join(source_dir, split, "labels", "bounding_boxes",
                                    annot_file), "r") as f:
                annot = json.load(f)

            annot["bounding_boxes"] = np.array(annot["bounding_boxes"])
            annot["labels"] = np.array(annot["labels"])
            deletes = np.where(annot["labels"] == 0)
            annot["bounding_boxes"] = np.delete(annot["bounding_boxes"], deletes, axis=0)
            annot["labels"] = np.delete(annot["labels"], deletes)

            yolo_file = img.split(".")[0] + ".txt"
            if os.path.exists(os.path.join(target_path, yolo_file)):
                os.remove(os.path.join(target_path, yolo_file))
            if len(annot["labels"]) > 0:
                with open(os.path.join(target_path, yolo_file), "w") as f:
                    for box, label in zip(annot["bounding_boxes"], annot["labels"]):
                        box = np.array(box, dtype=float)
                        new_box = box.copy()
                        new_box[:2] += (box[2:] - box[:2]) / 2
                        new_box[2:] -= box[:2]
                        new_box[0] /= w_orig
                        new_box[2] /= w_orig
                        new_box[1] /= h_orig
                        new_box[3] /= h_orig
                        line = [int(label)-1] + new_box.tolist()
                        line = [str(e) for e in line]
                        line = " ".join(line)
                        f.write(line)
                        f.write("\n")

print("Finished successfully.")