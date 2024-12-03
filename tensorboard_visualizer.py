import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


class TensorBoardVisualizer:
    def __init__(self):
        self.log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")

        # Use newer style setting
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

    def log_feature_embeddings(self, features, labels):
        """Create interactive 3D embedding visualization"""
        print("Creating feature embeddings...")

        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)

        # Create embeddings using TSNE
        tsne = TSNE(n_components=3, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(features_normalized)

        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(embeddings)

        # Create metadata as a list of lists
        metadata = [[str(label)] for label in labels]

        # Add embeddings to TensorBoard
        self.writer.add_embedding(
            embeddings_tensor,
            metadata=metadata,
            metadata_header=['Activity'],
            tag='Activity Features 3D'
        )

    def log_class_distribution(self, y_train, y_test):
        """Create interactive class distribution visualization"""
        print("Creating class distribution plots...")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Training set distribution
        sns.countplot(
            y=y_train,
            ax=ax1,
            order=np.unique(y_train)
        )
        ax1.set_title('Training Set Class Distribution')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

        # Test set distribution
        sns.countplot(
            y=y_test,
            ax=ax2,
            order=np.unique(y_test)
        )
        ax2.set_title('Test Set Class Distribution')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        self.writer.add_figure('Data_Distribution/Class_Balance', fig)
        plt.close()

    def log_feature_importance(self, feature_importance, feature_names=None):
        """Create interactive feature importance visualization"""
        print("Creating feature importance visualization...")

        # Get top 20 features
        n_features = 20
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]

        indices = np.argsort(feature_importance)[-n_features:]
        top_features = feature_importance[indices]
        top_names = [feature_names[i] for i in indices]

        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(n_features), top_features)

        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2,
                    f'{width:.4f}',
                    ha='left', va='center', fontsize=10)

        plt.yticks(range(n_features), top_names)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance Score')

        self.writer.add_figure('Feature_Analysis/Importance_Scores', fig)
        plt.close()

    def log_model_performances(self, results):
        """Log model performance metrics"""
        print("Creating model performance visualizations...")

        for model_name, result in results.items():
            metrics = result['metrics']

            # Log metrics
            self.writer.add_scalars(
                f'Performance/{model_name}',
                {
                    'Accuracy': metrics['accuracy'],
                    'F1_Score': metrics['f1_macro'],
                    'Precision': metrics['precision_macro'],
                    'Recall': metrics['recall_macro']
                }
            )

            # Add confusion matrix
            fig = plt.figure(figsize=(10, 8))
            sns.heatmap(
                result['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='YlOrRd'
            )
            plt.title(f'{model_name} Confusion Matrix')
            self.writer.add_figure(f'Confusion_Matrices/{model_name}', fig)
            plt.close()

    def close(self):
        self.writer.close()


def main():
    try:
        # Load data
        print("Loading data...")
        X_train = np.load('processed_data/X_train_features.npy')
        X_test = np.load('processed_data/X_test_features.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_test = np.load('processed_data/y_test.npy')

        # Initialize visualizer
        visualizer = TensorBoardVisualizer()

        print("Creating visualizations...")

        # Create embeddings
        visualizer.log_feature_embeddings(X_train, y_train)

        # Create class distribution plots
        visualizer.log_class_distribution(y_train, y_test)

        # Create feature importance visualization
        print("Training Random Forest for feature importance...")
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        visualizer.log_feature_importance(rf.feature_importances_)

        print("\nTensorBoard visualizations created successfully!")
        print("\nTo view the visualizations:")
        print(f"1. Run: tensorboard --logdir={visualizer.log_dir}")
        print("2. Open http://localhost:6006 in your browser")

        visualizer.close()

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()