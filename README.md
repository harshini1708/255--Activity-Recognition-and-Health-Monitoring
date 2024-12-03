# Project: Activity Recognition and Health Monitoring
#### CMPE-255
#### Group 13
#### Spring 2024
#### San Jose State University 
# OBJECTIVE
The objective of this project is to develop a robust and scalable system for recognizing human activities using wearable sensor data. By leveraging readings from accelerometers, gyroscopes, and magnetometers, the system classifies activities into daily and sports categories, detects anomalies, and provides actionable insights for fitness monitoring, rehabilitation, and elderly care. Advanced machine learning techniques were applied to ensure high classification accuracy and real-world applicability.
# Model
The system employs a Stacking Ensemble Model, combining the predictive power of Random Forest, XGBoost, and LightGBM models. The stacking architecture aggregates outputs from individual base models and passes them through a meta-model (Logistic Regression) for final predictions. This architecture is optimized for achieving high accuracy while being computationally efficient.

Additionally, individual models were evaluated to ensure the stacking approach provided superior performance:

1.Random Forest: Known for its reliability and interpretability.

2.XGBoost: Highly efficient and accurate, leveraging gradient boosting.

3.LightGBM: Optimized for speed and large datasets.

4.SVM: Effective for smaller datasets with high precision.

The stacking model achieved the highest accuracy of 99.74%, making it ideal for real-world applications.


# Usage
### Data Preparation
    data/
    ├── walking/
    │   ├── sensor1.txt
    │   ├── sensor2.txt
    ├── running/
    ├── cycling/

The dataset for this project contains sensor readings (accelerometer, gyroscope, magnetometer) sampled at 25 Hz, collected from 8 individuals performing 19 activities. Ensure the dataset follows the above structure

Dataset used: https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities
# Data Loading

The data loading process involves reading sensor data (accelerometer, gyroscope, and magnetometer readings) from directories organized by activity types. Each directory corresponds to a specific activity, such as walking, running, or cycling. The dataset is structured to facilitate easy access and mapping of sensor readings to their respective activity labels.

    import os
    import numpy as np

    def load_sensor_data(base_path):
        data = []
        labels = []
        activities = os.listdir(base_path)

        for activity in activities:
            activity_path = os.path.join(base_path, activity)
            files = os.listdir(activity_path)

            for file in files:
                file_path = os.path.join(activity_path, file)
                   # Load sensor readings from the file
                sensor_data = np.loadtxt(file_path)
               data.append(sensor_data)
               labels.append(activity)
    
        return np.array(data), np.array(labels)

    # Base path to the dataset
    base_path = "data/activities/"
    sensor_data, activity_labels = load_sensor_data(base_path)





## Data Preprocessing

The preprocessing steps involve:

Loading Data from Directories:
Sensor readings are loaded from folders organized by activity types (e.g., walking, running, cycling). Each folder contains files representing sensor data for a specific activity, with readings for x, y, and z axes.

Normalization:
To standardize sensor readings across all activities, each axis' values are normalized to have a mean of 0 and a standard deviation of 1. This step ensures consistency and improves model performance.

Segmentation:
Data is segmented into 2-second windows based on the 25 Hz sampling rate (50 samples per window). Each window captures meaningful activity patterns for downstream processing.

Feature Extraction:
Time-domain features (e.g., mean, standard deviation, RMS, skewness) and frequency-domain features (FFT for periodic components) are extracted for each segment. These features provide critical information about the activity being performed.



    from sklearn.preprocessing import StandardScaler

     # Normalize data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(sensor_readings)

     # Segment into 2-second windows
    segments = [normalized_data[i:i+50] for i in range(0, len(normalized_data), 50)]

# Feature Extraction
Feature extraction involves transforming raw sensor data into meaningful features that represent activity patterns effectively. This step is critical to improving the performance of machine learning models by focusing on the most relevant characteristics of the data.

### Steps in Feature Extraction

Time-Domain Features:

Extracted statistical metrics such as:

Mean: Average value of the sensor readings in a segment.

Standard Deviation: Measures variability in the data.

RMS (Root Mean Square): Captures energy within a segment.

Skewness: Asymmetry of the data distribution.

Kurtosis: Sharpness or flatness of the data distribution.

### Frequency-Domain Features:

Applied Fast Fourier Transform (FFT) to identify dominant frequencies, which are crucial for periodic activities like walking and running.
Calculated Spectral Mean, Spectral Standard Deviation, and Top Magnitudes from the FFT.

### Automated Feature Extraction:

Used a FeatureExtractor class to compute and organize these features for all data segments. This ensures consistent and efficient feature processing.

    from feature_extraction import FeatureExtractor

    extractor = FeatureExtractor()

    # Example: Extract features from a segment
    features = extractor.extract_features_from_segment(segment_data)

    # Feature matrix preparation
    feature_matrix = []
    for segment in segmented_data:
        feature_matrix.append(extractor.extract_features_from_segment(segment))

# Model Training and Testing
After feature extraction, the processed data is divided into training and testing sets to evaluate machine learning models. This step involves training multiple models and selecting the best-performing one for activity classification.

### Steps in Model Training and Testing
### Data Splitting:

The dataset is split into 80% training and 20% testing subsets using stratified sampling to maintain class balance.
Training data is used to build the model, while testing data evaluates its generalization performance.

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

# Model Selection:

Evaluated multiple machine learning models:

Random Forest: Reliable, interpretable, and robust.

XGBoost: High accuracy with gradient boosting.

LightGBM: Scalable and efficient for large datasets.

SVM: Effective for small datasets.

Used Stacking Ensemble as the final model for its superior accuracy, combining predictions from Random Forest, XGBoost, and LightGBM with Logistic Regression as the meta-model.

### Training the Model:

Models are trained on the training dataset with optimized hyperparameters.

    from sklearn.ensemble import StackingClassifier

    stack_model = StackingClassifier(
        estimators=[
             ('rf', RandomForestClassifier(n_estimators=100)),
             ('xgb', XGBClassifier(learning_rate=0.1, max_depth=6)),
             ('lgbm', LGBMClassifier())
        ],
        final_estimator=LogisticRegression()
    ) 

    stack_model.fit(X_train, y_train)

# Model Evaluation:

The model’s performance is evaluated using metrics such as:

Accuracy: Percentage of correctly classified activities.

Precision, Recall, and F1-Score: Measure the reliability and sensitivity of predictions.

ROC-AUC: Evaluates the trade-off between true positive rate and false positive rate.

    from sklearn.metrics import accuracy_score, classification_report

    y_pred = stack_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Summary 

This project successfully developed a robust and scalable system for activity recognition using wearable sensor data from accelerometers, gyroscopes, and magnetometers. By leveraging advanced data preprocessing techniques, statistical and frequency-based feature extraction, and machine learning models, the system achieves high accuracy and reliability.

The Stacking Ensemble Model demonstrated superior performance, achieving an accuracy of 99.74%, making it the most effective model for classifying activities across diverse individuals and scenarios. Supporting models like XGBoost and Random Forest also showcased high accuracy and interpretability.

This activity recognition system is ready for real-world applications in fitness monitoring, rehabilitation, and elderly care. Future enhancements could involve expanding the dataset to include more activities and implementing real-time monitoring capabilities for mobile and web platforms. The integration of this technology into daily life holds immense potential for improving health management and personalized fitness insights.

     












    


   
 

