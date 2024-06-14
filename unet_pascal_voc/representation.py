import joblib
import numpy as np
import cv2

def represent_data(img_dir, img_size):
    result = []
    images = joblib.load(img_dir)

    for image in images:
        new_images = cv2.resize(image, (img_size, img_size))
        result.append(new_images)

    result = np.array(result)
    return result

img_dir = "./data/data/pure_data/image.joblib"
mask_dir = "./data/data/pure_data/segmentation_class.joblib"
object_dir = "./data/data/pure_data/segmentation_object.joblib"

des_path = "./data/data/processed_data"
joblib.dump(represent_data(img_dir, 128), f"{des_path}/image.joblib")
joblib.dump(represent_data(mask_dir, 128), f"{des_path}/segmentation_class.joblib")
joblib.dump(represent_data(object_dir, 128), f"{des_path}/segmentation_object.joblib")