import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import joblib
import albumentations as A
from sklearn.metrics import f1_score

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "HardhatDetection-HOG-SVM-CV-FinalProject"
mlflow.set_experiment(experiment_name)

# User-defined run name
run_name = "HOG-RBF_SVC-RandomResample-NoFeatureSelector-Augmentation"

# Set Positive Label for dataset labeling
positive_label = "Hardhat"

# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)

# Dataset paths
dataset_path = r"D:\AI stuff\CV-LEC-FinalProject\Datasets\datasets2-cropped-64x64-hardhat-binary"

# Load annotations CSV
annotations_path = os.path.join(dataset_path, "annotations.csv")
annotations = pd.read_csv(annotations_path)

# Function to extract HOG features
def extract_hog_features(img, size=(64, 64)):
    img = img.resize(size)
    gray = img.convert('L')
    return hog(np.array(gray), orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)

# Function to load and augment images using annotations.csv
def load_images_from_annotations(annotations, split, augmentations=None):
    data, labels = [], []
    subset = annotations[annotations['data_split'] == split]
    label_map = {positive_label: 1, 'negative': 0}  # Map file_class to labels
    for _, row in tqdm(subset.iterrows(), desc=f"Processing {split} data", total=len(subset)):
        file_class = 'negative' if row['file_class'] == 'negative' else 'positive'
        img_path = os.path.join(dataset_path, file_class, row['file_name'])
        img = Image.open(img_path)
        label = label_map[row['file_class']]
        data.append(extract_hog_features(img))
        labels.append(label)
        if augmentations:
            augmented_img = augmentations(image=np.array(img))['image']
            augmented_img = Image.fromarray(augmented_img)
            data.append(extract_hog_features(augmented_img))
            labels.append(label)
    return data, labels

# Augmentation
augmentation = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.HueSaturationValue(p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
    A.ElasticTransform(p=0.2)
])

# Load datasets based on data_split
train_data, train_labels = load_images_from_annotations(annotations, "train", augmentation)
val_data, val_labels = load_images_from_annotations(annotations, "valid", augmentation)
test_data, test_labels = load_images_from_annotations(annotations, "test")

# Prepare data for model training
X = np.array(train_data + val_data)
y = np.array(train_labels + val_labels)

# Define scaler and feature selector
scaler = StandardScaler()

# K-Fold Cross-Validation
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_metrics = []

# Function to log class label distribution
def log_class_distribution(labels, stage=""):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"{stage} Class Distribution: {distribution}")
    mlflow.log_metrics({f"{stage}_class_{int(k)}_count": v for k, v in distribution.items()})


# Ensure any active run is ended
if mlflow.active_run():
    mlflow.end_run()

# Train and validate
with mlflow.start_run(run_name=run_name):
    # Log original class distribution
    log_class_distribution(train_labels + val_labels, stage="Original")

    # Resample data to balance classes
    X_resampled, y_resampled = resample(X, y, random_state=42, stratify=y)

    # Log resampled class distribution
    log_class_distribution(y_resampled, stage="Resampled")
    
    mlflow.log_params({
        "experiment_name": experiment_name,
        "dataset_path": dataset_path,
        "orientations": orientations,
        "pixels_per_cell": pixels_per_cell,
        "cells_per_block": cells_per_block,
        "k_folds": k_folds,
        "augmentation": augmentation.to_dict(),
        "resampling_technique": "Random" 
    })

    # Log dataset sizes
    mlflow.log_metrics({
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "resampled_train_size": len(X_resampled)
    })

    # Save preprocessing artifacts and models
    scaler_path = "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Log preprocessing artifacts to MLflow
    mlflow.log_artifact(scaler_path)

    # Save models for each fold and track in MLflow
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled)):
        X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
        y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = SVC(kernel='rbf')
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        fold_metrics.append(val_acc)

        # Save the model for the fold
        fold_model_path = f"model_fold_{fold + 1}.pkl"
        joblib.dump(model, fold_model_path)
        mlflow.log_artifact(fold_model_path)

        mlflow.log_metric(f"fold_{fold + 1}_val_accuracy", val_acc)

    # Save the final model after training on all folds
    final_model_path = "final_model.pkl"
    joblib.dump(model, final_model_path)
    mlflow.log_artifact(final_model_path)

    # Log mean validation accuracy
    mlflow.log_metric("mean_val_accuracy", np.mean(fold_metrics))

    # Final evaluation on the test set
    X_test = scaler.transform(test_data)
    test_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(test_labels, test_pred, target_names=['negative', positive_label])

    # Log test metrics and classification report
    test_acc = accuracy_score(test_labels, test_pred)
    cm = confusion_matrix(test_labels, test_pred)

    test_acc = accuracy_score(test_labels, test_pred)
    f1_positive = f1_score(test_labels, test_pred, pos_label=1)  # F1 score for positive class
    cm = confusion_matrix(test_labels, test_pred)

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("f1_positive", f1_positive)  # Log the F1 score for the positive class

    # Log classification report as an artifact
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Log confusion matrix plot
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['negative', positive_label], yticklabels=['negative', positive_label])
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Optionally, print the report to the console
    print(report)

    mlflow.end_run()