import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
import xgboost as xgb
import catboost as cb

# Function to extract HOG features
def extract_hog_features(img, size=(64, 128)):
    img = img.resize(size)
    gray = img.convert('L')
    return hog(np.array(gray), orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)

# Function to log class label distribution
def log_class_distribution(labels, stage=""):
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"{stage} Class Distribution: {distribution}")
    mlflow.log_metrics({f"{stage}_class_{int(k)}_count": v for k, v in distribution.items()})

# Parallel HOG Feature Extraction
def parallel_hog_features(row, dataset_path, label_map, img_size):
    file_class = 'negative' if row['file_class'] == 'negative' else 'positive'
    img_path = os.path.join(dataset_path, file_class, row['file_name'])
    
    try:
        img = Image.open(img_path)
    except (FileNotFoundError, IOError):
        return None, None

    label = label_map[row['file_class']]
    features = extract_hog_features(img, img_size)
    return features, label

# Extract features from annotations in parallel
def load_images_parallel(annotations, dataset_path, split, img_size):
    subset = annotations[annotations['data_split'] == split]
    label_map = {positive_label: 1, 'negative': 0}
    results = Parallel(n_jobs=-1)(
        delayed(parallel_hog_features)(row, dataset_path, label_map, img_size)
        for _, row in tqdm(subset.iterrows(), desc=f"Processing {split} data", total=len(subset))
    )

    # Filter None values
    data_labels = [item for item in results if item[0] is not None]
    data, labels = zip(*data_labels) if data_labels else ([], [])
    return data, labels

# HOG parameters
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)

# Dataset paths
master_dataset_path = r"D:\AI stuff\CV-LEC-FinalProject\Datasets\cropped-posneg-Construction Site Safety.v27-yolov8s.voc_2"

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
experiment_name = "CV-LEC-Final-Project-NewDataset-6/1/2025"
mlflow.set_experiment(experiment_name)

# Restrict to "Person" detection
positive_label = "Hardhat"

# Configurable parameters
resampling_strategy = False  # True to enable resampling
feature_selection = False  # True to enable feature selection
model_type = "LinearSVM"  # Options: 'LinearSVM', 'PolySVM', 'RbfSVM', 'XGBoost', 'CatBoost'
select_k_percentage = 0.8  # Percentage of features to select if feature selection is enabled

# Ensure only "Person" subdirectory is processed
for subdir in os.listdir(master_dataset_path):
    if subdir != positive_label:
        continue

    dataset_path = os.path.join(master_dataset_path, subdir)
    if not os.listdir(dataset_path):
        continue

    dataset_name = "cropped-Construction Site Safety.v27-yolov8s.voc_2"
    run_name = f"3-{positive_label}-{model_type}-{'Resample' if resampling_strategy else 'NoResample'}-{'FeatSel' if feature_selection else 'NoFeatSel'}"

    data_output_path = f'.\\ConstructionSafetyDataRuns\\{run_name}'
    os.makedirs(data_output_path, exist_ok=True)

    # Load annotations CSV
    annotations_path = os.path.join(dataset_path, "annotations.csv")
    annotations = pd.read_csv(annotations_path)

    img_size = (64, 128)

    # Parallelized HOG extraction
    train_data, train_labels = load_images_parallel(annotations, dataset_path, "train", img_size)
    val_data, val_labels = load_images_parallel(annotations, dataset_path, "valid", img_size)
    test_data, test_labels = load_images_parallel(annotations, dataset_path, "test", img_size)

    # Prepare data
    X = np.array(train_data + val_data)
    y = np.array(train_labels + val_labels)

    if resampling_strategy:
        X, y = resample(X, y, random_state=42, stratify=y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if feature_selection:
        num_features = int(X.shape[1] * select_k_percentage)
        selector = SelectKBest(score_func=f_classif, k=num_features)
        X = selector.fit_transform(X, y)

    # Define model
    if model_type == "LinearSVM":
        model = SVC(kernel='linear')
    elif model_type == "PolySVM":
        model = SVC(kernel='poly')
    elif model_type == "RbfSVM":
        model = SVC(kernel='rbf')
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier()
    elif model_type == "CatBoost":
        model = cb.CatBoostClassifier(verbose=0)
    else:
        raise ValueError("Invalid model type specified.")

    with mlflow.start_run(run_name=run_name):
        log_class_distribution(train_labels + val_labels, stage="Original")
        if resampling_strategy:
            log_class_distribution(y, stage="Resampled")

        # Train the model
        model.fit(X, y)

        # Test the model
        X_test = scaler.transform(test_data)
        if feature_selection:
            X_test = selector.transform(X_test)

        test_pred = model.predict(X_test)
        test_acc = accuracy_score(test_labels, test_pred)
        f1_positive = f1_score(test_labels, test_pred, pos_label=1)

        # Save the scaler
        scaler_path = os.path.join(data_output_path, f"{positive_label}-scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # Save the model
        model_path = os.path.join(data_output_path, f"{positive_label}-model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Log test results
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("f1_positive", f1_positive)

        # Classification report
        classification_report_path = os.path.join(data_output_path, f"{positive_label}-classification_report.txt")
        report = classification_report(test_labels, test_pred, target_names=['negative', positive_label])
        with open(classification_report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(classification_report_path)

        # Generate confusion matrix plot
        cm = confusion_matrix(test_labels, test_pred)

        plt.figure(figsize=(8, 6), dpi=150)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=['negative', positive_label],
            yticklabels=['negative', positive_label],
            square=True
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        cm_plot_path = os.path.join(data_output_path, f"{positive_label}-confusion_matrix.png")
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
