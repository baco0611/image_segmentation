import cv2
import joblib
import os

data_source_path = "../VOCdevkit/VOC2012/ImageSets/Segmentation"
image_source_path = "../VOCdevkit/VOC2012/JPEGImages"
segment_source_path = "../VOCdevkit/VOC2012/SegmentationObject"
segment_class_source_path = "../VOCdevkit/VOC2012/SegmentationClass"

with open(f'{data_source_path}/train.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

image_list = []
segment_list = []
segment_class_list = []

for line in lines:
    line = line.strip()
    print("Working on", line)

    image_path = os.path.join(image_source_path, f"{line}.jpg")
    segment_path = os.path.join(segment_source_path, f"{line}.png")
    segment_class_path = os.path.join(segment_class_source_path, f"{line}.png")
    
    image = cv2.imread(image_path)
    cv2.imwrite(f"./data/data/dataset/{line}.jpg", image)
    image_list.append(image)

    image_segment = cv2.imread(segment_path)
    cv2.imwrite(f"./data/data/SegmentationObject/{line}.jpg", image_segment)
    segment_list.append(image_segment)

    image_segment = cv2.imread(segment_class_path)
    cv2.imwrite(f"./data/data/SegmentationClass/{line}.jpg", image_segment)
    segment_class_list.append(image_segment)

data_path = "./data/data/pure_data"
joblib.dump(image_list, f"{data_path}/image.joblib")
joblib.dump(segment_list, f"{data_path}/segmentation_object.joblib")
joblib.dump(segment_class_list, f"{data_path}/segmentation_class.joblib")