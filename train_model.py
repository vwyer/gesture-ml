import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Database and model paths
DATABASE_PATH = 'hand_gestures.db'
MODEL_PATH = 'gesture_recognition_model.pkl'

def load_data():
    """Load data from the SQLite database and preprocess it for training."""
    conn = sqlite3.connect(DATABASE_PATH)
    query = "SELECT * FROM gesture_data"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Select relevant columns (all landmark columns)
    feature_columns = [col for col in df.columns if '_x' in col or '_y' in col]
    
    # Extract features and labels
    X = df[feature_columns].fillna(0)  # Fill missing values with 0 (for non-detected hands)
    y = df['gesture_type']

    # Print unique classes in the dataset
    print("Unique gesture types in the dataset:", y.unique())

    # Check class distribution
    print("Class distribution in the dataset:")
    print(y.value_counts())

    # Plot class distribution
    plt.figure(figsize=(6,4))
    y.value_counts().plot(kind='bar', color='skyblue')
    plt.title('Class Distribution')
    plt.xlabel('Gesture Type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.show()

    return X, y

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot a confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_learning_curve(model, X, y):
    """Plot learning curve to see how the model performance improves with more data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='accuracy', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(6,4))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()

def plot_3d_pca(X, y):
    """Perform PCA to reduce to 3 components and plot them in 3D."""
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Jede Klasse erhält eine eigene Farbe
    unique_classes = np.unique(y)
    colors = plt.cm.get_cmap('Set1', len(unique_classes))

    for i, cls in enumerate(unique_classes):
        indices = np.where(y == cls)
        ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], 
                   label=cls, color=colors(i), s=20, alpha=0.8)

    ax.set_title('3D PCA Cluster Analysis')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.legend()
    plt.tight_layout()
    plt.savefig('3d_pca_cluster.png')
    plt.show()

def train_model():
    """Train a gesture recognition model and produce plots for analysis."""
    # Load and preprocess data
    X, y = load_data()

    # Check if we have more than one class
    if len(y.unique()) <= 1:
        raise ValueError("The dataset must contain more than one unique class for training.")

    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize the Support Vector Machine (SVM) classifier
    model = SVC(kernel='linear', class_weight='balanced')

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print performance metrics
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, class_names=np.unique(y))

    # Plot Learning Curve with StratifiedKFold
    plot_learning_curve(SVC(kernel='linear', class_weight='balanced'), X, y)

    # Plot 3D PCA Cluster
    plot_3d_pca(X, y)

if __name__ == "__main__":
    train_model()
