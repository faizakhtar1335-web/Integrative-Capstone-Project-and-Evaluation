"""
Week 6 Integrative Capstone Project
Project: Handwritten Digit Classification using MNIST

Pipeline:
1. Data acquisition
2. Preprocessing
3. Exploratory data analysis
4. Supervised model building
5. Evaluation
6. Optional unsupervised analysis with PCA + K-Means
7. Insights and recommendations

Install:
    pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

Run:
    python week6_integrative_capstone_mnist.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = "week6_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load the publicly available MNIST handwritten-digit dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test


def preprocess_data(x_train, x_test):
    """Normalize grayscale pixels from 0-255 to 0-1."""
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


def exploratory_analysis(x_train, y_train):
    """Create basic EDA summaries and visualizations."""
    print("\n--- DATASET SUMMARY ---")
    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Image shape:", x_train.shape[1:])
    print("Number of classes:", len(np.unique(y_train)))

    label_counts = pd.Series(y_train).value_counts().sort_index()
    print("\nClass distribution:")
    print(label_counts)

    plt.figure(figsize=(8, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("MNIST Training Class Distribution")
    plt.xlabel("Digit")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i], cmap="gray")
        ax.set_title(f"Label: {y_train[i]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_digits.png"), dpi=150)
    plt.close()


def build_model():
    """Build a supervised neural-network classifier."""
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.20),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, x_train, y_train):
    """Train with a validation split and early stopping."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.10,
        epochs=15,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1,
    )

    return history


def plot_training_history(history):
    """Save accuracy and loss curves."""
    history_df = pd.DataFrame(history.history)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["accuracy"], label="Training Accuracy")
    plt.plot(history_df["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["loss"], label="Training Loss")
    plt.plot(history_df["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150)
    plt.close()


def evaluate_model(model, x_test, y_test):
    """Calculate classification metrics and save the confusion matrix."""
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

    print("\n--- MODEL EVALUATION ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    return accuracy, precision, recall, f1, predictions


def unsupervised_analysis(x_train, y_train, sample_size=5000):
    """
    Optional unsupervised component:
    PCA reduces the image vectors to 2 dimensions, then K-Means
    creates ten clusters. This is descriptive, not a supervised classifier.
    """
    x_flat = x_train.reshape(len(x_train), -1)

    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(x_flat), size=sample_size, replace=False)

    sample_x = x_flat[indices]
    sample_y = y_train[indices]

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(sample_x)

    kmeans = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(reduced)

    silhouette = silhouette_score(reduced, clusters)
    print("\n--- OPTIONAL UNSUPERVISED ANALYSIS ---")
    print(f"Silhouette score on PCA representation: {silhouette:.4f}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=clusters,
        s=8,
        alpha=0.6,
        cmap="tab10",
    )
    plt.colorbar(scatter, label="K-Means Cluster")
    plt.title("K-Means Clusters in 2D PCA Space")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_kmeans_clusters.png"), dpi=150)
    plt.close()

    cluster_table = pd.crosstab(
        pd.Series(clusters, name="Cluster"),
        pd.Series(sample_y, name="True_Digit"),
    )
    cluster_table.to_csv(os.path.join(OUTPUT_DIR, "cluster_digit_table.csv"))

    return silhouette


def main():
    print("Starting Week 6 Integrative Capstone Project...")

    x_train, y_train, x_test, y_test = load_data()

    # Preprocessing
    x_train, x_test = preprocess_data(x_train, x_test)

    # EDA
    exploratory_analysis(x_train, y_train)

    # Supervised learning
    model = build_model()
    model.summary()
    history = train_model(model, x_train, y_train)
    plot_training_history(history)

    accuracy, precision, recall, f1, predictions = evaluate_model(
        model, x_test, y_test
    )

    # Optional unsupervised analysis
    silhouette = unsupervised_analysis(x_train, y_train)

    model.save(os.path.join(OUTPUT_DIR, "mnist_digit_classifier.keras"))

    metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "Silhouette"],
            "Value": [accuracy, precision, recall, f1, silhouette],
        }
    )
    metrics.to_csv(os.path.join(OUTPUT_DIR, "evaluation_metrics.csv"), index=False)

    print("\nProject completed.")
    print(f"All generated files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
