# Activity Recognition and Health Monitoring Using Wearable Sensor Data

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Format](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

A state-of-the-art activity recognition system achieving up to 99.78% accuracy using ensemble learning and advanced feature engineering techniques. This project implements multiple machine learning models for classifying 19 different human activities using wearable sensor data.

## Team Members
- Harshini Pothireddy (017513548)
- Swetha Reddy Singireddy Damyella (017506554)
- Divyam Ashokbhai Savsaviya (017536714)

## Table of Contents
- [Project Overview](#project-overview)
- [Technical Details](#technical-details)
- [Dataset Information](#dataset-information)
- [System Architecture](#system-architecture)
- [Installation Guide](#installation-guide)
- [Usage Instructions](#usage-instructions)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Visualization Tools](#visualization-tools)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

### Objectives
- Implement real-time activity recognition system
- Develop robust feature extraction pipeline
- Compare and evaluate multiple ML approaches
- Create comprehensive visualization system
- Achieve >95% classification accuracy

### Key Features
- Multi-sensor data fusion
- Advanced feature extraction (time & frequency domain)
- Multiple ML model implementations
- Real-time processing capability
- Extensive visualization tools
- Cross-validation framework

## Technical Details

### Languages and Tools
- **Primary Language**: Python 3.8+
- **ML Frameworks**: 
  - scikit-learn 1.0.2
  - TensorFlow 2.8.0
  - XGBoost 1.5.0
  - LightGBM 3.3.2
  - CatBoost 1.0.6
- **Data Processing**: 
  - NumPy 1.21.0
  - Pandas 1.3.0
  - SciPy 1.7.0
- **Visualization**: 
  - Matplotlib 3.4.2
  - Seaborn 0.11.1
  - TensorBoard 2.8.0

### System Requirements
- **CPU**: Intel Core i5/AMD Ryzen 5 or better
- **RAM**: Minimum 16GB (32GB recommended)
- **Storage**: 10GB free space
- **GPU**: Optional, supports CUDA 11.0+
- **OS**: Ubuntu 20.04+/Windows 10+/macOS 10.15+

## Dataset Information

### Dataset Details
- **Name**: Daily and Sports Activities Dataset
- **Source**: UCI Machine Learning Repository
- **Size**: ~1.9 GB uncompressed
- **Instances**: 1,140,000 samples
- **Features per Sample**: 45 (9 features × 5 sensors)
- **Activities**: 19 different types
- **Subjects**: 8 individuals
- **Sampling Rate**: 25 Hz

### Activity Classes
1. Ascending stairs
2. Cycling (horizontal)
3. Cycling (vertical)
4. Descending stairs
5. Exercise cross trainer
6. Exercise stepper
7. Jumping
8. Lying (back)
9. Lying (right)
10. Moving elevator
11. Playing basketball
12. Rowing
13. Running (treadmill 8km/h)
14. Sitting
15. Standing
16. Standing (elevator still)
17. Walking (parking)
18. Walking (treadmill 4km/h)
19. Walking (treadmill 4km/h, 15° incline)

## Model Performance

### Comprehensive Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC AUC | Training Time (s) |
|-------|----------|-----------------|--------------|----------|----------|-------------------|
| LightGBM | 99.78% | 99.78% | 99.78% | 99.78% | 1.0000 | 55.41 |
| XGBoost | 99.71% | 99.71% | 99.71% | 99.71% | 1.0000 | 69.90 |
| Stacking | 99.74% | 99.75% | 99.74% | 99.74% | 1.0000 | 550.09 |
| CatBoost | 99.42% | 99.42% | 99.42% | 99.41% | 1.0000 | 95.05 |
| Random Forest | 99.12% | 99.13% | 99.12% | 99.12% | 0.9999 | 33.04 |
| Extra Trees | 99.23% | 99.25% | 99.23% | 99.23% | 0.9999 | 4.11 |
| SVM | 98.61% | 98.68% | 98.61% | 98.56% | 0.9999 | 101.48 |
| Neural Network | 98.28% | 98.29% | 98.28% | 98.28% | 0.9992 | 20.12 |
| KNN | 98.61% | 98.66% | 98.61% | 98.58% | 0.9978 | 0.57 |
| AdaBoost | 93.79% | 97.11% | 93.79% | 93.22% | 0.9998 | 1483.08 |

### Per-Activity Performance (LightGBM)
| Activity | Precision | Recall | F1-Score |
|----------|-----------|---------|-----------|
| Ascending stairs | 1.00 | 1.00 | 1.00 |
| Cycling horizontal | 1.00 | 1.00 | 1.00 |
| Cycling vertical | 1.00 | 1.00 | 1.00 |
| Descending stairs | 1.00 | 1.00 | 1.00 |
| Exercise cross trainer | 1.00 | 1.00 | 1.00 |
| Exercise stepper | 1.00 | 1.00 | 1.00 |
| Jumping | 1.00 | 1.00 | 1.00 |
| Lying back | 1.00 | 1.00 | 1.00 |
| Lying right | 1.00 | 1.00 | 1.00 |
| Moving elevator | 0.98 | 0.98 | 0.98 |
| Playing basketball | 1.00 | 0.98 | 0.99 |
| Rowing | 1.00 | 1.00 | 1.00 |
| Running treadmill 8km | 1.00 | 1.00 | 1.00 |
| Sitting | 1.00 | 1.00 | 1.00 |
| Standing | 1.00 | 1.00 | 1.00 |
| Standing elevator still | 0.98 | 1.00 | 0.99 |
| Walking parking | 1.00 | 1.00 | 1.00 |
| Walking treadmill 4km | 1.00 | 1.00 | 1.00 |
| Walking treadmill 15° | 1.00 | 1.00 | 1.00 |

## Project Structure
```
├── config/                 # Configuration files
│   ├── model_config.yaml   # Model hyperparameters
│   └── data_config.yaml    # Data processing parameters
├── data/
│   ├── processed/          # Processed dataset files
│   ├── raw/                # Raw sensor data
│   └── interim/            # Intermediate processing files
├── models/                 # Trained model files
│   ├── lightgbm/
│   ├── xgboost/
│   └── neural_networks/
├── notebooks/             # Jupyter notebooks
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── results/               # Evaluation results
│   ├── figures/
│   ├── metrics/
│   └── predictions/
├── src/
│   ├── data/             # Data processing scripts
│   │   ├── preprocess.py
│   │   └── validate.py
│   ├── features/         # Feature engineering
│   │   ├── extractors.py
│   │   └── selectors.py
│   ├── models/          # Model implementations
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/   # Visualization tools
│       ├── plots.py
│       └── dashboard.py
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
├── setup.py            # Package setup
└── README.md           # Project documentation
```

## Installation Guide

### Prerequisites
1. Python environment setup:
```bash
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. Clone repository:
```bash
git clone https://github.com/Pruthvik-Reddy/PR_Project_Sem5.git
cd PR_Project_Sem5
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional GPU Support
For GPU acceleration:
```bash
pip install tensorflow-gpu
pip install torch torchvision torchaudio cudatoolkit=11.0
```

## Usage Instructions

### Data Preprocessing
```bash
# Process raw data
python src/data/preprocess_data.py --input_dir data/raw --output_dir data/processed

# Extract features
python src/features/extract_features.py --input_dir data/processed --output_dir data/interim
```

### Model Training
```bash
# Train specific model
python src/models/train.py --model lightgbm --config config/model_config.yaml

# Train all models
python src/models/train_all.py --config config/model_config.yaml
```

### Evaluation
```bash
# Evaluate single model
python src/models/evaluate.py --model lightgbm --data data/processed/test.csv

# Compare all models
python src/models/compare_models.py --results_dir results/metrics
```

### Visualization
```bash
# Launch TensorBoard
tensorboard --logdir=runs

# Generate performance plots
python src/visualization/generate_plots.py --results_dir results/metrics
```

## Contributing
Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact
For technical queries:
- Harshini Pothireddy - [GitHub](https://github.com/harshini)
- Swetha Reddy Singireddy Damyella - [GitHub](https://github.com/swetha)
- Divyam Ashokbhai Savsaviya - [GitHub](https://github.com/divyam)

Project Repository: [https://github.com/Pruthvik-Reddy/PR_Project_Sem5](https://github.com/Pruthvik-Reddy/PR_Project_Sem5)
