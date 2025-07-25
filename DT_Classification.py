import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
)
from imblearn.over_sampling import SMOTE
from collections import Counter

# --- Configuration ---
DATASET_CONFIGS = {
    "1": {
        "name": "LWSNDR Single Hop Indoor",
        "file": "LWSNDR Single Hop Indoor.csv",
    },
    "2": {
        "name": "LWSNDR Multi Hop Indoor",
        "file": "LWSNDR Multi Hop Indoor.csv",
    },
    "3": {
        "name": "satellite",
        "file": "satellite.csv",
    },
    "4": {
        "name": "IoT_23_data",
        "file": "IoT_23_Data.csv",
    },
}

BASE_DIR = "/home/ubuntu/ids-lab/"
CLUSTERED_DATA_DIR = f"{BASE_DIR}Datasets/clustered_Dataset"

# Random seeds for reproducible results across multiple runs
RANDOM_SEEDS = [10, 100, 1000, 2000, 3500]


def get_user_choice():
    """Prompts the user to select a dataset and returns the corresponding config."""
    print("Please Enter the data that you want to use:\n\nOptions Are:")
    print("----------------------------------------------------")
    for key, config in DATASET_CONFIGS.items():
        print(f"Enter {key} for {config['name']}")
    print("----------------------------------------------------")

    while True:
        option = input("Your choice: ")
        if option in DATASET_CONFIGS:
            return DATASET_CONFIGS[option]
        print("Invalid Option. Please try again.")


def load_clustered_data(config):
    """Load the clustered dataset based on the configuration."""
    file_path = f"{CLUSTERED_DATA_DIR}/{config['file']}"
    try:
        data = pd.read_csv(file_path)
        print(f"Successfully loaded: {config['name']}")
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None


def analyze_class_balance(data):
    """Analyze and display the class balance in the dataset."""
    class_counts = Counter(data["ClusterLabel"])
    print("Class Balance Analysis:")
    print("----------------------------------------------------")
    for label, count in class_counts.items():
        print(f"Class '{label}': {count} samples")
    print("----------------------------------------------------")
    print("Note: 'anomaly' = anomaly class, 'normal' = normal class\n")
    return class_counts


def prepare_data_for_training(data):
    """Prepare the data for training by applying SMOTE oversampling."""
    # Convert the label to categorical
    data = data.copy()
    data["ClusterLabel"] = data["ClusterLabel"].astype("category")

    # Calculate the oversampling ratio to be within the defined limits
    majority_class_count = data["ClusterLabel"].value_counts().max()
    minority_class_count = data["ClusterLabel"].value_counts().min()
    over_ratio = (
        0.1 * majority_class_count - minority_class_count
    ) / minority_class_count
    over_ratio = min(over_ratio, 1.0)

    print(f"Calculated SMOTE sampling strategy: {over_ratio:.4f}")

    # Separate features and labels - exclude both ClusterLabel and Label columns
    columns_to_exclude = ["ClusterLabel"]
    if "Label" in data.columns:
        columns_to_exclude.append("Label")

    X = data.drop(columns=columns_to_exclude)
    y = data["ClusterLabel"]

    # Ensure all feature columns are numeric
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(
            f"Warning: Found non-numeric columns that will be excluded: {non_numeric_cols}"
        )
        X = X.select_dtypes(include=[np.number])

    print(f"Original data shape - X: {X.shape}, y: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")

    # Apply SMOTE for oversampling
    smote = SMOTE(
        sampling_strategy=over_ratio,
        k_neighbors=min(5, minority_class_count - 1),
        random_state=10,
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Display results after SMOTE
    print(f"After SMOTE - X: {X_resampled.shape}, y: {y_resampled.shape}")
    print("Class distribution after SMOTE:")
    resampled_counts = Counter(y_resampled)
    for label, count in resampled_counts.items():
        print(f"  Class '{label}': {count} samples")
    print()

    return X_resampled, y_resampled, y


def train_and_evaluate_model(X_resampled, y_resampled, original_y):
    """Train and evaluate the Decision Tree model across multiple random seeds."""
    # Initialize metrics storage
    metrics = {"fpr": [], "f1_score": [], "precision": [], "recall": [], "pr_auc": []}

    print("Training Decision Tree models with different random seeds...")
    print("=" * 60)

    # Train the model multiple times with different seeds
    for i, seed in enumerate(RANDOM_SEEDS, 1):
        print(f"Training iteration {i}/{len(RANDOM_SEEDS)} (seed: {seed})")
        np.random.seed(seed)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled,
            y_resampled,
            test_size=0.3,
            random_state=seed,
            stratify=y_resampled,
        )

        # Train the decision tree
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Create confusion matrix
        conf_mat = confusion_matrix(y_test, y_pred, labels=original_y.cat.categories)

        # Calculate classification metrics
        class_report = classification_report(
            y_test, y_pred, labels=original_y.cat.categories, output_dict=True
        )

        # Store macro-averaged metrics
        metrics["f1_score"].append(class_report["macro avg"]["f1-score"])
        metrics["precision"].append(class_report["macro avg"]["precision"])
        metrics["recall"].append(class_report["macro avg"]["recall"])

        # Calculate False Positive Rate
        tn, fp, fn, tp = conf_mat.ravel()
        if (fp + tn) > 0:
            fpr_value = fp / (fp + tn)
            metrics["fpr"].append(fpr_value)
        else:
            metrics["fpr"].append(np.nan)

        # Calculate Area Under Precision-Recall Curve (AUCPR)
        y_pred_proba = clf.predict_proba(X_test)

        # Get the class labels in the order used by the classifier
        class_labels = clf.classes_

        # Find the index of the "anomaly" class
        anomaly_idx = list(class_labels).index("anomaly")

        # Get probability of the anomaly class
        y_scores_anomaly = y_pred_proba[:, anomaly_idx]

        precision, recall, _ = precision_recall_curve(
            y_test == "anomaly", y_scores_anomaly, pos_label=True
        )
        pr_auc_score = auc(recall, precision)
        metrics["pr_auc"].append(pr_auc_score)

    print("Training completed!\n")
    return metrics


def display_results(metrics):
    """Display the evaluation results in a formatted manner."""
    os.system("clear")

    print("=" * 65)
    print("DECISION TREE CLASSIFICATION RESULTS")
    print("=" * 65)
    print()

    # Define metric display information
    metric_info = [
        ("False Positive Rate", "fpr"),
        ("F-measure", "f1_score"),
        ("Sensitivity (Recall)", "recall"),
        ("Precision", "precision"),
        ("Area Under Precision-Recall Curve (AUCPR)", "pr_auc"),
    ]

    # Display each metric with mean and standard deviation
    for display_name, metric_key in metric_info:
        values = metrics[metric_key]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values, ddof=1)

        print("=" * 65)
        print(f"{display_name}")
        print("-" * 65)
        print(f"Mean: {mean_val:.6f}")
        print(f"Standard Deviation: {std_val:.6f}")
        print("=" * 65)
        print()


def main():
    """Main function to run the Decision Tree classification workflow."""
    # Set random seed for reproducibility
    np.random.seed(10)

    # 1. Get user input and load data
    config = get_user_choice()
    clustered_data = load_clustered_data(config)

    if clustered_data is None:
        print("Failed to load data. Exiting.")

    # 2. Analyze class balance
    analyze_class_balance(clustered_data)

    # 3. Prepare data for training (apply SMOTE)
    X_resampled, y_resampled, original_y = prepare_data_for_training(clustered_data)

    # 4. Train and evaluate the model
    metrics = train_and_evaluate_model(X_resampled, y_resampled, original_y)

    # 5. Display results
    display_results(metrics)


if __name__ == "__main__":
    main()
