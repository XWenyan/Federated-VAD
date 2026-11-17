import os
import cv2

def convert_jpg_to_png(root_dir):
    """
    Recursively convert all .jpg files under root_dir to .png
    Keep original jpg files.
    """

    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                jpg_path = os.path.join(root, file)
                png_path = os.path.join(
                    root,
                    os.path.splitext(file)[0] + ".png"
                )

                if os.path.exists(png_path):
                    continue   # already converted

                img = cv2.imread(jpg_path)
                os.remove(jpg_path)
                if img is None:
                    print("Failed to read:", jpg_path)
                    continue

                cv2.imwrite(png_path, img)
                count += 1

    print(f"Done. Converted {count} images.")


if __name__ == "__main__":
    ped2_root = "/mnt/private/xwy/aed-mae/data/ped2"  
    convert_jpg_to_png(ped2_root)
