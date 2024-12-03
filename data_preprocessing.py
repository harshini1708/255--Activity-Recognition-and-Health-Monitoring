import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class DataLoader:
    def __init__(self, data_path: str):
        print(f"Initializing DataLoader with path: {data_path}")
        self.data_path = data_path

        # Initialize activity labels
        self.activity_labels = {
            'a01': 'sitting',
            'a02': 'standing',
            'a03': 'lying_back',
            'a04': 'lying_right',
            'a05': 'ascending_stairs',
            'a06': 'descending_stairs',
            'a07': 'standing_elevator_still',
            'a08': 'moving_elevator',
            'a09': 'walking_parking',
            'a10': 'walking_treadmill_4km',
            'a11': 'walking_treadmill_4km_15deg',
            'a12': 'running_treadmill_8km',
            'a13': 'exercising_stepper',
            'a14': 'exercising_cross_trainer',
            'a15': 'cycling_horizontal',
            'a16': 'cycling_vertical',
            'a17': 'rowing',
            'a18': 'jumping',
            'a19': 'playing_basketball'
        }

        self.sensors = ['torso', 'right_arm', 'left_arm', 'right_leg', 'left_leg']
        self.measurements = ['acc_x', 'acc_y', 'acc_z',
                             'gyro_x', 'gyro_y', 'gyro_z',
                             'mag_x', 'mag_y', 'mag_z']

        self.columns = []
        for sensor in self.sensors:
            for measurement in self.measurements:
                self.columns.append(f"{sensor}_{measurement}")

    def load_segment(self, file_path: str) -> np.ndarray:
        """
        Load a single data segment from file

        Args:
            file_path: Path to data file

        Returns:
            numpy array with sensor readings
        """
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Process each line
            data = []
            for line in lines:
                # Split the line by commas and convert to float
                values = [float(x.strip()) for x in line.strip().split(',')]
                data.append(values)

            return np.array(data)
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None

    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        total_segments = 0

        print(f"Starting to load data from {self.data_path}")

        # Iterate through activity folders (a01-a19)
        for activity in self.activity_labels.keys():
            activity_path = os.path.join(self.data_path, activity)
            if not os.path.exists(activity_path):
                print(f"Warning: Activity path does not exist: {activity_path}")
                continue

            participant_folders = sorted([d for d in os.listdir(activity_path)
                                          if os.path.isdir(os.path.join(activity_path, d))])

            print(f"Activity {activity}: Found {len(participant_folders)} participants")

            for participant in participant_folders:
                participant_path = os.path.join(activity_path, participant)
                segment_files = sorted(glob(os.path.join(participant_path, "s*.txt")))
                print(f"  Participant {participant}: Found {len(segment_files)} segments")

                for file_path in segment_files:
                    segment_data = self.load_segment(file_path)
                    if segment_data is not None:
                        X.append(segment_data)
                        y.append(self.activity_labels[activity])
                        total_segments += 1

        print(f"\nTotal segments loaded: {total_segments}")
        if total_segments == 0:
            raise ValueError("No data was loaded! Please check the data directory structure.")

        return np.array(X), np.array(y)


class DataPreprocessor:
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()

    def preprocess_data(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Preprocess the raw sensor data

        Args:
            X: Raw sensor data of shape (n_segments, n_timesteps, n_features)
            fit: Whether to fit the scaler on this data

        Returns:
            Preprocessed data
        """
        print(f"Input shape: {X.shape}")  # Debug print

        # Reshape to 2D for scaling
        n_segments, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(n_segments * n_timesteps, n_features)

        print(f"Reshaped for scaling: {X_reshaped.shape}")  # Debug print

        # Scale the data
        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)

        # Reshape back to 3D
        X_preprocessed = X_scaled.reshape(n_segments, n_timesteps, n_features)

        print(f"Output shape: {X_preprocessed.shape}")  # Debug print

        return X_preprocessed


def load_and_preprocess_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Main function to load and preprocess all data
    """
    # Load data
    loader = DataLoader(data_path)
    X, y = loader.load_all_data()

    print(f"\nLoaded data shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of features per timestep: {X.shape[2]}")

    # Split into train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("\nPreprocessing data...")
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train = preprocessor.preprocess_data(X_train, fit=True)
    X_test = preprocessor.preprocess_data(X_test, fit=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = "/Users/harshinireddy/Downloads/dataset_dm"

    print("\nChecking data path structure...")
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        exit(1)

    print("\nLoading and preprocessing data...")
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

        # Create output directory if it doesn't exist
        os.makedirs('processed_data', exist_ok=True)

        print("\nFinal Results:")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")

        # Save processed data
        print("\nSaving processed data...")
        save_path = 'processed_data'
        np.save(os.path.join(save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)
        print(f"Data saved successfully in {save_path}/ directory!")

        # Print some basic statistics
        print("\nClass distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"{label}: {count} samples")

    except Exception as e:
        print(f"\nError occurred during processing: {str(e)}")
        import traceback

        traceback.print_exc()
