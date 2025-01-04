import numpy as np
import cv2
import joblib
from imutils.object_detection import non_max_suppression
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt
import Sliding as sd
import mlflow
import io
import tempfile

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Sliding window parameters
size = (64, 64)
step_size = (9, 9)
downscale = 1.25

# List to store the detections
detections = []
scale = 0

# Load the image
# image_path = r"C:\Users\willi\Downloads\istockphoto-1319979559-612x612.jpg" 
# image_path = r"C:\Users\willi\Downloads\360557d1-53a9-457b-838d-eefcd42f3634.__CR275,0,4449,3337_PT0_SX600_V1___.jpg"  
# image_path = r"C:\Users\willi\Downloads\trimble-mixed-reality-hard-hat.webp"  
# image_path = r"C:\Users\willi\Downloads\WhatsApp Image 2025-01-02 at 20.19.10_730bca96.jpg"  #no hardhat
image_path = r"C:\Users\willi\Downloads\What_type_of_person_are_you_quiz_pic.png"  #no hardhat

image = cv2.imread(image_path)
image = cv2.resize(image, (400, 256))

# Load the pre-trained model and scaler (SVM)
model_path = r"D:\AI stuff\CV-LEC-FinalProject\mlartifacts\382771492849991826\40a284c8e3ce42d084fc99c45efb573e\artifacts\final_model.pkl"
scaler_path = r"D:\AI stuff\CV-LEC-FinalProject\mlartifacts\844529703615738274\2e1ac46a514649faa70d54ead1738963\artifacts\scaler.pkl"  
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ef529a1910a242079b84ab7f2f7249c5 negative face data try 1
# try 2 dacddd265843428aa9333731dd7bb254
# resample, nofeature, aug 7ad327c29d1748fb8d96984cd014c130
# resample, nofeature, aug, DS2 40a284c8e3ce42d084fc99c45efb573e

# Start a new MLflow run
experiment_name = "HardhatDetection-HOG-SVM-CV-FinalProject"
run_name = "VIS-DS2-HOG-SVM-RandomResample-NoFeatureSelector-Augmentation"
mlflow.set_experiment(experiment_name)

# Get the experiment ID
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Search for existing runs with the same name
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.mlflow.runName = '{run_name}'"
)

def generate_heatmap(image, detections):
    """
    Generate a heatmap based on the detections.

    :param image: Original image
    :param detections: List of detected bounding boxes [(x1, y1, score, w, h), ...]
    :return: Heatmap
    """
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    for x, y, _, w, h in detections:
        heatmap[y:y + h, x:x + w] += 1  # Increment heatmap in bounding box area
    
    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Apply a threshold to the heatmap to remove false positives.

    :param heatmap: Heatmap generated from detections
    :param threshold: Threshold value
    :return: Thresholded heatmap
    """
    heatmap[heatmap <= threshold] = 0
    return heatmap


def detection():
    # Process the image with a sliding window at multiple scales using Gaussian pyramid
    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        # Stop when the image is smaller than the window size
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break

        # Apply sliding window on the scaled image
        for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue

            # Convert window to grayscale and extract HOG features
            window = color.rgb2gray(window)
            fd = hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2')

            # Rescale features using the scaler
            fd = scaler.transform([fd])

            # Classify the feature vector using the trained model
            pred = model.predict(fd)
            score = model.decision_function(fd)[0]  # Get the decision function score for SVM
            # print(f"Score: {score}")
            
            if pred == 1 and score >= 3.5:  # If prediction is positive (Hardhat detected)
                detections.append(
                    (int(x * (downscale**scale)), int(y * (downscale**scale)), score,
                     int(size[0] * (downscale**scale)), int(size[1] * (downscale**scale))))

        scale += 1

    # Generate heatmap from detections
    heatmap = generate_heatmap(image, detections)

    # Apply threshold to the heatmap
    threshold = 50  # Adjust threshold as needed
    heatmap = apply_threshold(heatmap, threshold)

    # # Visualize the heatmap
    # plt.figure()
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heatmap with Threshold')
    # plt.colorbar()
    # plt.show()

    # Draw the raw detections before applying non-max suppression
    clone = image.copy()
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = np.array([score for (x, y, score, w, h) in detections])

    # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

    # Draw the final bounding boxes after NMS
    for (x1, y1, x2, y2) in pick:
        cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(clone, 'Hardhat', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Visualize the final image with bounding boxes
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title('Detected Hardhats')
    plt.axis('off')
    plt.show()

    # Log the detection results to MLflow
    _, img_encoded = cv2.imencode('.jpg', clone)
    img_bytes = io.BytesIO(img_encoded.tobytes())
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file_path = temp_file.name
        cv2.imwrite(temp_file_path, clone)  # Write the image to the temporary file
        mlflow.log_artifact(temp_file_path, artifact_path="output_image.jpg")

    mlflow.log_param("image_path", image_path)
    mlflow.log_metric("detections", len(pick))

# List to store scores from decision_function
scores = []

def detection2():
    global scores  # Declare scores as global to modify it
    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break

        for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue

            window = color.rgb2gray(window)
            fd = hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2')
            fd = scaler.transform([fd])

            pred = model.predict(fd)
            score = model.decision_function(fd)[0]  # Decision function score
            scores.append(score)  # Append score to the list

            if pred == 1 and score >= 5:
                detections.append(
                    (int(x * (downscale**scale)), int(y * (downscale**scale)), score,
                     int(size[0] * (downscale**scale)), int(size[1] * (downscale**scale))))
        scale += 1

    # Plot the distribution of scores
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Decision Function Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Save scores to a file for further analysis
    with open('decision_scores.txt', 'w') as f:
        for score in scores:
            f.write(f"{score}\n")




# If a matching run is found, get its run_id
if runs:
    run_id = runs[0].info.run_id
    # Start MLflow run (either new or existing)
    with mlflow.start_run(run_id=run_id, nested=True):
        detection()
else:
    run_id = None
    with mlflow.start_run(run_name=run_name, nested=True):
        detection()