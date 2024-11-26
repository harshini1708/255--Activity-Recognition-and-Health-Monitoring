import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  # Add this import
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os


class ModelEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(y_train)
        self.y_test = self.label_encoder.transform(y_test)

        self.results = {}
        self.cv_folds = 5

        # Create results directory
        os.makedirs('results/model_performance', exist_ok=True)

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive set of metrics"""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, recall, F1 (macro and weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred,
                                                                                              average='weighted')

        metrics.update({
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        })

        # Per-class metrics
        classes = np.unique(y_true)
        per_class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None, labels=classes)

        metrics['per_class'] = {
            'precision': per_class_metrics[0],
            'recall': per_class_metrics[1],
            'f1': per_class_metrics[2],
            'support': per_class_metrics[3]
        }

        # ROC AUC if probability predictions available
        if y_pred_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr')
            except:
                metrics['roc_auc'] = None

        return metrics

    def train_and_evaluate(self, model_name, model, save_plots=True):
        print(f"\nTraining {model_name}...")
        start_time = time.time()

        try:
            # Train and cross-validate
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv_folds)

            # Train final model
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(self.X_test)
            try:
                y_pred_proba = model.predict_proba(self.X_test)
            except:
                y_pred_proba = None

            # Convert predictions to original labels
            y_test_original = self.label_encoder.inverse_transform(self.y_test)
            y_pred_original = self.label_encoder.inverse_transform(y_pred)

            # Calculate metrics
            metrics = self.calculate_metrics(y_test_original, y_pred_original, y_pred_proba)
            conf_matrix = confusion_matrix(y_test_original, y_pred_original)
            report = classification_report(y_test_original, y_pred_original)

            # Store results
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'cv_scores': cv_scores,
                'training_time': train_time,
                'confusion_matrix': conf_matrix,
                'report': report
            }

            # Print results
            self._print_results(model_name)

            # Generate plots
            if save_plots:
                self._save_plots(model_name)

        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            raise

    def train_all_models(self):
        # Define base models for stacking
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42))
        ]

        # Define meta-classifier for stacking
        meta_classifier = LogisticRegression(random_state=42)

        # Define all models
        models = {
            'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'CatBoost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_seed=42, verbose=False),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
            'Stacking': StackingClassifier(
                estimators=base_models,
                final_estimator=meta_classifier,
                cv=5
            )
        }

        # Train and evaluate each model
        for name, model in models.items():
            try:
                self.train_and_evaluate(name, model)
            except Exception as e:
                print(f"Skipping {name} due to error: {str(e)}")
                continue
    def _print_results(self, model_name):
        """Print comprehensive results"""
        result = self.results[model_name]
        metrics = result['metrics']

        print(f"\n{model_name} Results:")
        print("-" * 50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nMacro-averaged metrics:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall: {metrics['recall_macro']:.4f}")
        print(f"  F1-score: {metrics['f1_macro']:.4f}")
        print(f"\nWeighted-averaged metrics:")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall: {metrics['recall_weighted']:.4f}")
        print(f"  F1-score: {metrics['f1_weighted']:.4f}")
        print(
            f"\nCross-validation scores (mean ± std): {result['cv_scores'].mean():.4f} ± {result['cv_scores'].std():.4f}")
        print(f"Training time: {result['training_time']:.2f} seconds")
        if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
            print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(result['report'])

    def _save_plots(self, model_name):
        """Generate and save all plots"""
        result = self.results[model_name]

        # Confusion Matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(f'results/model_performance/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

        # Per-class metrics
        self._plot_per_class_metrics(model_name)

    def _plot_per_class_metrics(self, model_name):
        """Plot detailed per-class metrics"""
        metrics = self.results[model_name]['metrics']['per_class']
        classes = self.label_encoder.classes_

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        fig.suptitle(f'Per-class Performance Metrics - {model_name}')

        # Plot precision, recall, and F1 separately
        for idx, (metric_name, metric_values) in enumerate([
            ('Precision', metrics['precision']),
            ('Recall', metrics['recall']),
            ('F1-Score', metrics['f1'])
        ]):
            axes[idx].bar(classes, metric_values)
            axes[idx].set_title(f'Per-class {metric_name}')
            axes[idx].set_xticklabels(classes, rotation=45, ha='right')
            axes[idx].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(f'results/model_performance/per_class_metrics_{model_name.lower().replace(" ", "_")}.png')
        plt.close()

    def plot_comparative_results(self):
        """Plot comparative results for all models"""
        if not self.results:
            print("No results to plot!")
            return

        models = list(self.results.keys())
        metrics = {
            'Accuracy': [self.results[m]['metrics']['accuracy'] for m in models],
            'Macro F1': [self.results[m]['metrics']['f1_macro'] for m in models],
            'Training Time': [self.results[m]['training_time'] for m in models],
            'CV Score': [self.results[m]['cv_scores'].mean() for m in models]
        }

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison')

        for (metric_name, metric_values), ax in zip(metrics.items(), axes.ravel()):
            ax.bar(models, metric_values)
            ax.set_title(f'{metric_name} Comparison')
            ax.set_xticklabels(models, rotation=45, ha='right')

            # Add value labels on top of bars
            for i, v in enumerate(metric_values):
                ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('results/model_performance/model_comparison.png')
        plt.close()

    def save_results(self):
        """Save detailed results to file"""
        if not self.results:
            print("No results to save!")
            return

        with open('results/model_performance/detailed_results.txt', 'w') as f:
            for name, result in self.results.items():
                f.write(f"\n{'-' * 50}\n")
                f.write(f"{name} Results:\n")
                f.write(f"Accuracy: {result['metrics']['accuracy']:.4f}\n")
                f.write(f"Macro Precision: {result['metrics']['precision_macro']:.4f}\n")
                f.write(f"Macro Recall: {result['metrics']['recall_macro']:.4f}\n")
                f.write(f"Macro F1: {result['metrics']['f1_macro']:.4f}\n")
                if 'roc_auc' in result['metrics'] and result['metrics']['roc_auc'] is not None:
                    f.write(f"ROC AUC Score: {result['metrics']['roc_auc']:.4f}\n")
                f.write(f"Training Time: {result['training_time']:.2f} seconds\n")
                f.write("\nClassification Report:\n")
                f.write(result['report'])
                f.write("\n")


def main():
    try:
        # Load features
        print("Loading extracted features...")
        X_train = np.load('processed_data/X_train_features.npy')
        X_test = np.load('processed_data/X_test_features.npy')
        y_train = np.load('processed_data/y_train.npy')
        y_test = np.load('processed_data/y_test.npy')

        print("\nData shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")

        # Initialize evaluator
        evaluator = ModelEvaluator(X_train, X_test, y_train, y_test)

        # Train and evaluate all models
        evaluator.train_all_models()

        # Generate comparative plots
        print("\nGenerating comparative plots...")
        evaluator.plot_comparative_results()

        # Save detailed results
        print("\nSaving detailed results...")
        evaluator.save_results()

        print("\nModel training and evaluation completed successfully!")
        print("Check 'results/model_performance' directory for detailed outputs.")

    except Exception as e:
        print(f"\nError during model evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()