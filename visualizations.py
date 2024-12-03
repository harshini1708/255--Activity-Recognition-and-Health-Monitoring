# visualizations.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os


class DataVisualizer:
    def __init__(self):
        # Create visualization directories
        os.makedirs('results/visualizations', exist_ok=True)
        os.makedirs('results/visualizations/data_analysis', exist_ok=True)
        os.makedirs('results/visualizations/feature_analysis', exist_ok=True)
        os.makedirs('results/visualizations/model_analysis', exist_ok=True)

    def load_data(self):
        """Load all preprocessed and results data"""
        print("Loading data...")
        # Load preprocessed data
        self.X_train = np.load('processed_data/X_train.npy')
        self.X_test = np.load('processed_data/X_test.npy')
        self.y_train = np.load('processed_data/y_train.npy')
        self.y_test = np.load('processed_data/y_test.npy')

        # Load feature data
        self.X_train_features = np.load('processed_data/X_train_features.npy')
        self.X_test_features = np.load('processed_data/X_test_features.npy')

        print("Data shapes:")
        print(f"X_train: {self.X_train.shape}")
        print(f"X_train_features: {self.X_train_features.shape}")

    def create_data_distribution_plots(self):
        """Create visualizations for data distribution"""
        print("Creating data distribution plots...")

        plt.figure(figsize=(15, 10))

        # Class distribution
        plt.subplot(2, 2, 1)
        unique_classes, counts = np.unique(self.y_train, return_counts=True)
        plt.bar(range(len(unique_classes)), counts)
        plt.title('Class Distribution in Training Set')
        plt.xlabel('Activity Class')
        plt.ylabel('Count')

        # Feature distribution (using feature matrix)
        plt.subplot(2, 2, 2)
        # Take mean across time steps for first 5 sensors
        sensor_means = np.mean(self.X_train, axis=1)[:, :5]
        plt.boxplot([sensor_means[:, i] for i in range(5)])
        plt.title('Sensor Distribution (First 5 Sensors)')
        plt.xlabel('Sensor Index')
        plt.ylabel('Mean Value')

        # Feature importance (using extracted features)
        plt.subplot(2, 2, 3)
        plt.hist(self.X_train_features[:, 0], bins=50)
        plt.title('Distribution of First Extracted Feature')
        plt.xlabel('Feature Value')
        plt.ylabel('Count')

        # Time series plot for first sensor
        plt.subplot(2, 2, 4)
        plt.plot(self.X_train[0, :, 0])
        plt.title('Time Series Plot (First Sample, First Sensor)')
        plt.xlabel('Time Step')
        plt.ylabel('Sensor Value')

        plt.tight_layout()
        plt.savefig('results/visualizations/data_analysis/data_distribution.png')
        plt.close()

    def create_feature_analysis_plots(self):
        """Create feature analysis visualizations"""
        print("Creating feature analysis plots...")

        # Feature correlation heatmap (first 10 features)
        plt.figure(figsize=(12, 8))
        corr_matrix = np.corrcoef(self.X_train_features[:, :10].T)
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True)
        plt.title('Feature Correlation Matrix (First 10 Features)')
        plt.savefig('results/visualizations/feature_analysis/feature_correlations.png')
        plt.close()

        # Feature importance using random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X_train_features, self.y_train)

        plt.figure(figsize=(12, 6))
        importance = rf.feature_importances_
        indices = np.argsort(importance)[-20:]
        plt.barh(range(20), importance[indices])
        plt.yticks(range(20), [f'Feature {i}' for i in indices])
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('results/visualizations/feature_analysis/feature_importance.png')
        plt.close()

    def create_activity_analysis_plots(self):
        """Create activity-specific analysis plots"""
        print("Creating activity analysis plots...")

        # Time series patterns for different activities
        unique_activities = np.unique(self.y_train)[:4]  # Take first 4 activities
        plt.figure(figsize=(15, 10))

        for i, activity in enumerate(unique_activities):
            activity_data = self.X_train[self.y_train == activity]
            plt.subplot(2, 2, i + 1)
            plt.plot(activity_data[0, :, 0])  # Plot first sample, first sensor
            plt.title(f'Activity: {activity}')
            plt.xlabel('Time Step')
            plt.ylabel('Sensor Value')

        plt.tight_layout()
        plt.savefig('results/visualizations/data_analysis/activity_patterns.png')
        plt.close()

    def create_sensor_correlation_plots(self):
        """Create sensor correlation visualizations"""
        print("Creating sensor correlation plots...")

        # Calculate mean sensor values
        sensor_means = np.mean(self.X_train, axis=1)  # Mean across time steps

        # Correlation between first 5 sensors
        plt.figure(figsize=(10, 8))
        corr_matrix = np.corrcoef(sensor_means[:, :5].T)
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True)
        plt.title('Sensor Correlation Matrix (First 5 Sensors)')
        plt.savefig('results/visualizations/data_analysis/sensor_correlations.png')
        plt.close()


def main():
    visualizer = DataVisualizer()
    visualizer.load_data()

    # Create all visualizations
    visualizer.create_data_distribution_plots()
    visualizer.create_feature_analysis_plots()
    visualizer.create_activity_analysis_plots()
    visualizer.create_sensor_correlation_plots()

    print("\nAll visualizations have been created successfully!")
    print("Check the 'results/visualizations' directory for all plots.")


if __name__ == "__main__":
    main()