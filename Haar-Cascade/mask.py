# Import library
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize label, classifier, actuals, and predictions
class_label = os.listdir('test/mask')
mask_classifier = cv2.CascadeClassifier('cascade_classifier/mask.xml')

y_true = []
y_pred = []

# Iterate from test folder
for index, label in enumerate(class_label):
    label_dir = os.path.join('test/mask', label)
    for idx, image_name in enumerate(os.listdir(label_dir)):
        print(f"Running on {idx}")
        image_path = os.path.join(label_dir, image_name)
        
        # Preprocess the image to be gray
        image_bgr = cv2.imread(image_path)
        image_gray = cv2.imread(image_path, 0)
        
        # Skip if there is no person detected
        if image_gray is None:
            continue
        
        # Detect
        detected_mask = mask_classifier.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=5)
        
        # Input to actuals
        y_true.append(label)

        # Input to predictions
        if len(detected_mask) < 1:
            y_pred.append('n')
            continue
        else:
            y_pred.append('p')

# Visualize accuracy metrics and confusion matrix    
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.show()