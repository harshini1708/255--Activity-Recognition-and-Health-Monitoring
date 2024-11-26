# feature_extraction.py

import numpy as np
from scipy import stats
from scipy.fft import fft
import os
from typing import Dict, List, Tuple
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor"""
        print("Initializing Feature Extractor...")

    def extract_statistical_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from time series data
        """
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'var': np.var(data),
            'max': np.max(data),
            'min': np.min(data),
            'median': np.median(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'range': np.ptp(data),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'energy': np.sum(np.square(data)) / len(data)
        }
        return features

    def extract_frequency_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT
        """
        # Compute FFT
        fft_coeffs = fft(data)
        fft_coeffs = fft_coeffs[:len(data) // 2]

        # Magnitude spectrum
        magnitudes = np.abs(fft_coeffs)

        # Get top 5 frequencies and their magnitudes
        top_indices = np.argsort(magnitudes)[-5:]

        features = {}
        for i, idx in enumerate(top_indices):
            features[f'fft_mag_{i + 1}'] = magnitudes[idx]
            features[f'fft_freq_{i + 1}'] = idx

        # Add spectral features
        features.update({
            'spectral_mean': np.mean(magnitudes),
            'spectral_std': np.std(magnitudes),
            'spectral_energy': np.sum(np.square(magnitudes)) / len(magnitudes)
        })

        return features

    def extract_features_from_segment(self, segment: np.ndarray) -> np.ndarray:
        """
        Extract features from a single segment
        """
        all_features = []

        # For each sensor measurement
        for i in range(segment.shape[1]):
            time_series = segment[:, i]

            # Extract features
            stat_features = self.extract_statistical_features(time_series)
            freq_features = self.extract_frequency_features(time_series)

            # Combine all features
            features = list(stat_features.values()) + list(freq_features.values())
            all_features.extend(features)

        return np.array(all_features)

    def analyze_features(self, X_features: np.ndarray, y: np.ndarray):
        """
        Analyze extracted features
        """
        print("\nFeature Analysis:")
        print("-----------------")

        # Basic Statistics
        print("\nFeature Statistics:")
        print(f"Number of features: {X_features.shape[1]}")
        print(f"Number of samples: {X_features.shape[0]}")

        # Check for NaN or Infinite values
        nan_count = np.isnan(X_features).sum()
        inf_count = np.isinf(X_features).sum()
        print(f"\nData Quality:")
        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")

        # Feature Importance using Mutual Information
        print("\nCalculating feature importance...")
        mi_scores = mutual_info_classif(X_features, y)

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

        # Plot top 20 most important features
        plt.figure(figsize=(12, 6))
        top_features_idx = np.argsort(mi_scores)[-20:]
        plt.bar(range(20), mi_scores[top_features_idx])
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Mutual Information Score')
        plt.savefig('results/feature_importance.png')
        plt.close()

        # Feature Correlation Analysis
        print("\nCalculating feature correlations...")
        sample_features = X_features[:, top_features_idx]
        corr_matrix = np.corrcoef(sample_features.T)

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix (Top 20 Features)')
        plt.savefig('results/feature_correlations.png')
        plt.close()

        return {
            'mi_scores': mi_scores,
            'correlation_matrix': corr_matrix,
            'top_features_idx': top_features_idx
        }


def extract_all_features(X: np.ndarray, print_progress: bool = True) -> np.ndarray:
    """
    Extract features from all segments
    """
    extractor = FeatureExtractor()
    features_list = []
    n_segments = X.shape[0]

    for i in range(n_segments):
        if print_progress and (i + 1) % 100 == 0:
            print(f"Processing segment {i + 1}/{n_segments}")

        segment_features = extractor.extract_features_from_segment(X[i])
        features_list.append(segment_features)

    return np.array(features_list)


if __name__ == "__main__":
    print("Loading preprocessed data...")

    try:
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_path = os.path.join(current_dir, 'processed_data')
        results_path = os.path.join(current_dir, 'results')

        # Verify and create directories
        for directory in [processed_data_path, results_path]:
            if not os.path.exists(directory):
                print(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)

        # Verify input files exist
        required_files = ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']
        for file in required_files:
            file_path = os.path.join(processed_data_path, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load preprocessed data
        print("\nLoading preprocessed data files...")
        X_train = np.load(os.path.join(processed_data_path, 'X_train.npy'))
        X_test = np.load(os.path.join(processed_data_path, 'X_test.npy'))
        y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(processed_data_path, 'y_test.npy'))

        print("\nData shapes before feature extraction:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        # Validate data shapes
        if len(X_train.shape) != 3 or len(X_test.shape) != 3:
            raise ValueError("Input data must be 3-dimensional (samples, timesteps, features)")

        # Extract features
        print("\nExtracting features from training data...")
        X_train_features = extract_all_features(X_train)

        print("\nExtracting features from test data...")
        X_test_features = extract_all_features(X_test)

        print("\nData shapes after feature extraction:")
        print(f"X_train_features shape: {X_train_features.shape}")
        print(f"X_test_features shape: {X_test_features.shape}")

        # Validate extracted features
        if np.isnan(X_train_features).any() or np.isnan(X_test_features).any():
            raise ValueError("NaN values detected in extracted features")

        # Perform feature analysis
        print("\nAnalyzing features...")
        extractor = FeatureExtractor()
        analysis_results = extractor.analyze_features(X_train_features, y_train)

        # Save features with error handling
        print("\nSaving extracted features...")
        try:
            np.save(os.path.join(processed_data_path, 'X_train_features.npy'), X_train_features)
            np.save(os.path.join(processed_data_path, 'X_test_features.npy'), X_test_features)
        except Exception as e:
            raise IOError(f"Error saving feature files: {str(e)}")

        # Print summary statistics
        print("\nFeature extraction summary:")
        print("---------------------------")
        print(f"Total features extracted per segment: {X_train_features.shape[1]}")
        print(f"Total segments processed: {X_train_features.shape[0] + X_test_features.shape[0]}")

        # Print top features
        print("\nTop 5 most important features:")
        mi_scores = analysis_results['mi_scores']
        top_features = analysis_results['top_features_idx'][-5:]
        for idx in top_features:
            print(f"Feature {idx}: MI Score = {mi_scores[idx]:.4f}")

        print("\nFeature extraction completed successfully!")
        print(f"Features saved in: {processed_data_path}")
        print(f"Analysis plots saved in: {results_path}")

    except FileNotFoundError as e:
        print(f"\nError: Required input files not found: {str(e)}")
        print("Please ensure the preprocessed data files exist in the correct location.")
    except ValueError as e:
        print(f"\nError: Invalid data format: {str(e)}")
        print("Please check the input data shapes and values.")
    except Exception as e:
        print(f"\nUnexpected error during feature extraction: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
    finally:
        print("\nFeature extraction process finished.")