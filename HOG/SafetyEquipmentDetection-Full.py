import numpy as np
import cv2
import joblib
import Sliding as sd
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt

# Load the image
# image_path = r"C:\Users\willi\Downloads\images.jpg"
image_path = r"C:\Users\willi\Downloads\istockphoto-1319979559-612x612.jpg" 
# image_path = r"C:\Users\willi\Downloads\360557d1-53a9-457b-838d-eefcd42f3634.__CR275,0,4449,3337_PT0_SX600_V1___.jpg"  
# image_path = r"C:\Users\willi\Downloads\trimble-mixed-reality-hard-hat.webp"  
# image_path = r"C:\Users\willi\Downloads\WhatsApp Image 2025-01-02 at 20.19.10_730bca96.jpg"  #no hardhat
# image_path = r"C:\Users\willi\Downloads\What_type_of_person_are_you_quiz_pic.png"  #no hardhat
image = cv2.imread(image_path)
image = cv2.resize(image, (400, 256))

# Sliding window parameters
helmet_window_size = (64, 64)  # Hardhat detection window
person_window_size = (64, 128)  # Person detection window
step_size = (9, 9)
downscale = 1.25

# List to store the detections
detections = []
scale = 0

# hardhat_model_path = r"D:\AI stuff\CV-LEC-FinalProject\mlartifacts\382771492849991826\0b41a581ccff46f58710d31c2a7c1da0\artifacts\final_model.pkl"
# person_model_path = r"D:\AI stuff\CV-LEC-FinalProject\HOG\PersonDetection\models\models.dat"
# scaler_path = r"D:\AI stuff\CV-LEC-FinalProject\mlartifacts\844529703615738274\2e1ac46a514649faa70d54ead1738963\artifacts\scaler.pkl"

hardhat_model_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Hardhat-LinearSVM-Random-AugOn\Hardhat-final_model.pkl"
hardhat_scaler_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Hardhat-LinearSVM-Random-AugOn\Hardhat-scaler.pkl"
person_model_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Person-LinearSVM-Random-AugOn\Person-final_model.pkl"
person_scaler_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Person-LinearSVM-Random-AugOn\Person-scaler.pkl"
vest_model_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Safety Vest-LinearSVM-Random-AugOn\Safety Vest-final_model.pkl"
vest_scaler_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Safety Vest-LinearSVM-Random-AugOn\Safety Vest-scaler.pkl"
mask_model_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Mask-LinearSVM-Random-AugOn\Mask-final_model.pkl"
mask_scaler_path = r"D:\AI stuff\CV-LEC-FinalProject\ConstructionSafetyDataRuns\3-Mask-LinearSVM-Random-AugOn\Mask-scaler.pkl"

# Load the pre-trained models
hardhat_model = joblib.load(hardhat_model_path)
hardhat_scaler = joblib.load(hardhat_scaler_path)
person_model = joblib.load(person_model_path)
person_scaler = joblib.load(person_scaler_path)
vest_model = joblib.load(vest_model_path)
vest_scaler = joblib.load(vest_scaler_path)
mask_model = joblib.load(mask_model_path)
mask_scaler = joblib.load(mask_scaler_path)

# Add mask and vest detection
for im_scaled in pyramid_gaussian(image, downscale=downscale):
    if im_scaled.shape[0] < person_window_size[1] or im_scaled.shape[1] < person_window_size[0]:
        break

    for (x, y, window) in sd.sliding_window(im_scaled, person_window_size, step_size):
        if window.shape[0] != person_window_size[1] or window.shape[1] != person_window_size[0]:
            continue

        window = color.rgb2gray(window)
        fd = hog(window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        fd = fd.reshape(1, -1)
        fd_person_scaled = person_scaler.transform(fd)

        pred = person_model.predict(fd_person_scaled)

        if pred == 1 and person_model.decision_function(fd)[0] >= 0.2:
            person_crop = im_scaled[y:y + person_window_size[1], x:x + person_window_size[0]]
            person_crop_resized = cv2.resize(person_crop, helmet_window_size)
            person_crop_gray = color.rgb2gray(person_crop_resized)

            # Hardhat Detection
            fd_hardhat = hog(person_crop_gray, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
            fd_hardhat = fd_hardhat.reshape(1, -1)
            fd_hardhat_scaled = hardhat_scaler.transform(fd_hardhat)
            hardhat_pred = hardhat_model.predict(fd_hardhat_scaled)
            hardhat_confidence = hardhat_model.decision_function(fd_hardhat_scaled)[0]

            # Mask Detection
            fd_mask_scaled = mask_scaler.transform(fd_hardhat)
            mask_pred = mask_model.predict(fd_mask_scaled)

            # Vest Detection
            fd_vest_scaled = vest_scaler.transform(fd_hardhat)
            vest_pred = vest_model.predict(fd_vest_scaled)

            # Labels based on detections
            label = []
            if hardhat_pred == 1 and hardhat_confidence >= 0.5:
                label.append("Hardhat")
            else:
                label.append("No Hardhat")
            
            if mask_pred == 1:
                label.append("Mask")
            else:
                label.append("No Mask")
            
            if vest_pred == 1:
                label.append("Vest")
            else:
                label.append("No Vest")
            
            label_text = ', '.join(label)

            # Box color: green if all present, red if any missing
            if "No" in label_text:
                box_color = 'red'
            else:
                box_color = 'green'

            detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), box_color, int(person_window_size[0] * (downscale**scale)), int(person_window_size[1] * (downscale**scale)), label_text))

    scale += 1

# Draw the raw detections before applying non-max suppression
clone = image.copy()
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h, _) in detections])
colors = [box_color for (_, _, box_color, _, _, _) in detections]

# Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
pick = non_max_suppression(rects, overlapThresh=0.3)

# Draw the final bounding boxes
for i, (x1, y1, x2, y2) in enumerate(pick):
    box_color = colors[i]
    color_rgb = (0, 255, 0) if box_color == 'green' else (0, 0, 255)
    cv2.rectangle(clone, (x1, y1), (x2, y2), color_rgb, 2)
    cv2.putText(clone, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)


plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Person Detection with Safety Equipment Check")
plt.show()


# Optionally, save the output image
# cv2.imwrite(r'D:\AI stuff\CV-LEC-FinalProject\HOG\PersonHardhatDetection\image_out.png', clone)
