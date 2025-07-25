import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, adjusted_rand_score
import matplotlib.pyplot as plt

import seaborn as sns

# --- Configuration ---
# Store dataset-specific settings in a dictionary to avoid repetitive code.
DATASET_CONFIGS = {
    "1": {
        "name": "LWSNDR Multi Hop Indoor",
        "file": "LWSNDR Multi Hop Indoor.csv",
        "x_col": "Humidity",
        "y_col": "Temperature",
    },
    "2": {
        "name": "LWSNDR Single Hop Indoor",
        "file": "LWSNDR Single Hop Indoor.csv",
        "x_col": "Humidity",
        "y_col": "Temperature",
    },
    "3": {
        "name": "satellite",
        "file": "satellite.csv",
        "x_col": "V1",
        "y_col": "V2",
    },
    "4": {
        "name": "IoT_23_data",
        "file": "IoT_23_data.csv",
        "x_col": "Index",
        "y_col": "duration",
    },
}

# Directories used to retrieve and
BASE_DIR = "/home/ubuntu/ids-lab/"
DATA_DIR = f"{BASE_DIR}Datasets/original_Dataset"
RESULTS_DIR = f"{BASE_DIR}Graphs"
CLUSTERED_DATA_DIR = f"{BASE_DIR}Datasets/clustered_Dataset"


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


def main():
    """Main function to run the clustering analysis workflow."""
    # 1. Get user input and set up paths
    config = get_user_choice()
    dataset_name = config["name"]

    # Create output directories if they don't exist
    results_path = f"{RESULTS_DIR}/{dataset_name}"
    import os

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(CLUSTERED_DATA_DIR, exist_ok=True)

    print()
    # 2. Load and Prepare Data
    file = config["file"]
    file_path = f"{DATA_DIR}/{file}"
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return

    # Separate features (X) and true labels (y)
    features = data.iloc[:, :-1]
    true_labels = data.iloc[:, -1]

    # Your original script calls .astype(float) but doesn't assign the result.
    # Pandas methods that modify a DataFrame often return a new one.
    # You must assign the result back to a variable.
    features = features.astype(float)

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # 3. Initial Visualization
    plt.figure(figsize=(8, 6), dpi=300)
    sns.scatterplot(x=data[config["x_col"]], y=data[config["y_col"]], hue=true_labels)
    plt.title(f"Original Data: {dataset_name}")
    plt.legend(title="Original Labels")
    plt.savefig(f"{results_path}/original.tiff", dpi=300)
    plt.close()  # Close plot to free memory

    # 4. Perform Clustering
    # Your script performed clustering twice. This consolidates it to one method.
    agg_clustering = AgglomerativeClustering(
        n_clusters=2, metric="manhattan", linkage="average"
    )
    predicted_labels_numeric = agg_clustering.fit_predict(normalized_features)

    # Convert to string labels for visualization and saving
    predicted_labels_str = [
        "anomaly" if label == 1 else "normal" for label in predicted_labels_numeric
    ]

    # 5. Evaluate and Visualize Clustering Results
    # --- Silhouette Analysis ---
    # Use numeric labels for silhouette analysis
    silhouette_avg = np.mean(
        silhouette_samples(normalized_features, predicted_labels_numeric)
    )

    plt.figure(figsize=(8, 6), dpi=300)
    # Custom plotting logic can be complex; libraries like yellowbrick simplify this.
    # For now, we'll keep a simplified version of your original plot.
    # This example assumes 2 clusters as defined in AgglomerativeClustering
    x_lower = 10
    unique_labels = np.unique(predicted_labels_numeric)
    silhouette_vals = silhouette_samples(normalized_features, predicted_labels_numeric)
    for i, cluster_label in enumerate(unique_labels):
        cluster_silhouette_vals = silhouette_vals[
            predicted_labels_numeric == cluster_label
        ]
        cluster_silhouette_vals.sort()
        size_cluster_i = cluster_silhouette_vals.shape[0]
        x_upper = x_lower + size_cluster_i

        color = sns.color_palette("husl", len(unique_labels))[i]
        plt.fill_between(
            np.arange(x_lower, x_upper),
            0,
            cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        x_lower = x_upper + 10

    plt.axhline(y=silhouette_avg, color="red", linestyle="--")
    plt.ylabel("Silhouette coefficient values")
    plt.xlabel("Cluster label")
    plt.title(
        f"Silhouette Plot for {dataset_name}\nScore: {silhouette_avg:.3f}", fontsize=12
    )
    plt.savefig(f"{results_path}/Silhouette.tiff", dpi=300)
    plt.close()

    # --- Adjusted Rand Index ---
    # The original script calculated this but never used it. Now we print it.
    # Robustly handle non-numeric true labels before scoring.
    cleaned_true_labels = (
        pd.to_numeric(true_labels, errors="coerce").fillna(-1).astype(int)
    )
    if -1 in cleaned_true_labels.unique():
        print(
            "Warning: Non-numeric ground truth labels found and ignored in ARI calculation."
        )

    adjusted_rand = adjusted_rand_score(cleaned_true_labels, predicted_labels_numeric)
    print(f"Adjusted Rand Index: {adjusted_rand:.4f}")

    # --- HAP Visualization ---
    plt.figure(figsize=(8, 6), dpi=300)
    # Create a custom color palette: normal = blue, anomaly = red
    custom_palette = {"normal": "#56B4E9", "anomaly": "red"}
    sns.scatterplot(
        x=data[config["x_col"]],
        y=data[config["y_col"]],
        hue=predicted_labels_str,
        palette=custom_palette,
    )
    plt.title(f"{dataset_name} - Clustered Results")
    plt.legend(title="Cluster Type")
    plt.savefig(f"{results_path}/Clustered.tiff", dpi=300)
    plt.close()

    # 6. Save Clustered Data
    # Your original method for saving the CSV was destructive.
    # It blanked out all column headers. This is the correct way to add a new column.
    output_data = data.copy()
    output_data["ClusterLabel"] = predicted_labels_str
    output_path = f"{CLUSTERED_DATA_DIR}/{dataset_name}.csv"
    output_data.to_csv(output_path, index=False)

    print("\nClustering Completed.")
    print(f"Results saved in: {results_path}")
    print(f"Clustered data saved to: {output_path}")


if __name__ == "__main__":
    main()
