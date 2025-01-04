import cv2
import xml.etree.ElementTree as ET
import pathlib
import csv
import os
from tqdm import tqdm  # Import tqdm for progress tracking

def read_voc_xml(xml_path):
    """Parse an XML file in PASCAL VOC format."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ann = {"filename": root.find("filename").text, "objects": []}
    for obj in root.findall("object"):
        obj_dict = {
            "name": obj.find("name").text,
            "xmin": int(obj.find("bndbox/xmin").text),
            "xmax": int(obj.find("bndbox/xmax").text),
            "ymin": int(obj.find("bndbox/ymin").text),
            "ymax": int(obj.find("bndbox/ymax").text)
        }
        ann["objects"].append(obj_dict)
    return ann

def make_square(xmin, xmax, ymin, ymax):
    """Shrink the bounding box to square shape."""
    xcenter = (xmax + xmin) // 2
    ycenter = (ymax + ymin) // 2
    halfdim = min(xmax - xmin, ymax - ymin) // 2
    xmin, xmax = xcenter - halfdim, xcenter + halfdim
    ymin, ymax = ycenter - halfdim, ycenter + halfdim
    return xmin, xmax, ymin, ymax

def is_overlap(xmin, xmax, ymin, ymax, bxmin, bxmax, bymin, bymax):
    """Check if two bounding boxes overlap."""
    return not (xmax <= bxmin or xmin >= bxmax or ymax <= bymin or ymin >= bymax)

def crop_and_save_images(base_path, output_dir, positive_label, winSize=(64, 64)):
    """Crop images based on bounding boxes and save them to output_dir, separating positive and negative samples."""
    positive_dir = pathlib.Path(output_dir) / "positive"
    negative_dir = pathlib.Path(output_dir) / "negative"
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pathlib.Path(output_dir) / "annotations.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['index', 'file_name', 'file_class', 'data_split']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        index = 0
        for split in ["train", "valid", "test"]:
            ann_src = pathlib.Path(base_path) / split
            xml_files = list(ann_src.glob("*.xml"))
            for xmlfile in tqdm(xml_files, desc=f"Processing {split} split", unit="file"):
                ann = read_voc_xml(str(xmlfile))
                image_path = ann_src / ann["filename"]
                img = cv2.imread(str(image_path))
                
                if img is None:
                    continue
                
                img_h, img_w = img.shape[:2]
                positive_bboxes = []

                for obj in ann["objects"]:
                    xmin, xmax, ymin, ymax = obj["xmin"], obj["xmax"], obj["ymin"], obj["ymax"]
                    xmin, xmax, ymin, ymax = make_square(xmin, xmax, ymin, ymax)
                    
                    if obj["name"] == positive_label:
                        positive_bboxes.append((xmin, xmax, ymin, ymax))
                        cropped_img = img[ymin:ymax, xmin:xmax]
                        
                        if cropped_img.size == 0:
                            continue
                        
                        cropped_img = cv2.resize(cropped_img, winSize)
                        file_name = f"{index}_{obj['name']}.jpg"
                        cv2.imwrite(str(positive_dir / file_name), cropped_img)
                        
                        writer.writerow({
                            'index': index, 
                            'file_name': file_name, 
                            'file_class': obj['name'], 
                            'data_split': split
                        })
                        index += 1

                # Slide 64x64 window across the image for negative samples
                step_x, step_y = winSize
                for y in range(0, img_h - step_y + 1, step_y):
                    for x in range(0, img_w - step_x + 1, step_x):
                        overlap = False
                        for (bxmin, bxmax, bymin, bymax) in positive_bboxes:
                            if is_overlap(x, x + step_x, y, y + step_y, bxmin, bxmax, bymin, bymax):
                                overlap = True
                                break
                        
                        if not overlap:
                            cropped_img = img[y:y + step_y, x:x + step_x]
                            file_name = f"{index}_negative.jpg"
                            cv2.imwrite(str(negative_dir / file_name), cropped_img)
                            
                            writer.writerow({
                                'index': index, 
                                'file_name': file_name, 
                                'file_class': "negative", 
                                'data_split': split
                            })
                            index += 1

# Define parameters
base_path = pathlib.Path(r"D:\AI stuff\CV-LEC-FinalProject\Datasets\Person detection.v6i.voc")
output_dir = pathlib.Path(r"D:\AI stuff\CV-LEC-FinalProject\Datasets\Person detection.v6i-cropped-64x128")
positive_label = "Persona"  # Specify the positive label here

# Crop images and save
crop_and_save_images(base_path, output_dir, positive_label, winSize=(64, 128))
