import glob
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif"]

def compute_gradients(data_root_folder, step, folder):

    extension = None
    for ext in IMG_EXTENSIONS:
        if len(list(glob.glob(os.path.join(data_root_folder, f"{folder}/frames", f"*/*{ext}")))) > 0:
            extension = ext
            break

    dirs = list(glob.glob(os.path.join(data_root_folder, folder, "frames", "*")))

    for video in tqdm(dirs):
        img_paths = list(glob.glob(os.path.join(video, f"*{extension}")))
        img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))

        for i, img_path in enumerate(img_paths):

            previous = max(i - step, 0)
            next = min(i + step, len(img_paths) - 1)

            # ---- read as grayscale ----
            previous_img = cv2.imread(img_paths[previous], cv2.IMREAD_GRAYSCALE)
            next_img = cv2.imread(img_paths[next], cv2.IMREAD_GRAYSCALE)

            previous_img = previous_img.astype(np.int32)
            next_img = next_img.astype(np.int32)

            gradient = np.abs(previous_img - next_img)
            gradient = gradient.astype(np.uint8)

            # ---- convert to 3 channels ----
            gradient = np.expand_dims(gradient, -1).repeat(3, axis=-1)

            save_dir = os.path.join(
                data_root_folder,
                f"{folder}/gradients2/{os.path.basename(video)}"
            )
            os.makedirs(save_dir, exist_ok=True)

            Image.fromarray(gradient).save(
                os.path.join(save_dir, os.path.basename(img_path))
            )


if __name__ == "__main__":

    root_folder_ped2 = "/mnt/private/xwy/aed-mae/data/ped2"
    compute_gradients(root_folder_ped2, 1, "train")
    compute_gradients(root_folder_ped2, 1, "test")
