"""
Model training and evaluation utilities for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, accuracy_score
)
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    """Model trainer for Customer Churn Prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.model_scores = {}
        
    def initialize_models(self):
        """Initialize base models with default parameters"""
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'xgboost': XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                enable_categorical=False,
                use_label_encoder=False
            )
        }
        return self.models
    
    def get_hyperparameter_grids(self):
        """Define hyperparameter grids for each model"""
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        return param_grids
    
    def train_base_models(self, X_train, y_train, cv_folds=5):
        """Train base models without hyperparameter tuning"""
        print("Training base models...")
        
        self.initialize_models()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Fit the model
                model.fit(X_train, y_train)
                
                # Cross-validation scores with error handling for XGBoost
                if name == 'xgboost':
                    # Use a more compatible approach for XGBoost
                    cv_scores = []
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        
                        # Create a fresh model for this fold
                        fold_model = XGBClassifier(
                            random_state=42, 
                            eval_metric='logloss',
                            enable_categorical=False,
                            use_label_encoder=False
                        )
                        fold_model.fit(X_fold_train, y_fold_train)
                        y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
                        score = roc_auc_score(y_fold_val, y_pred_proba)
                        cv_scores.append(score)
                    cv_scores = np.array(cv_scores)
                else:
                    # Standard cross-validation for other models
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                self.model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores
                }
                
                print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
                print(f"Skipping {name} and continuing with other models...")
                continue
        
        return self.models, self.model_scores
    
    def hyperparameter_tuning(self, X_train, y_train, cv_folds=3, scoring='roc_auc'):
        """Perform hyperparameter tuning for all models"""
        print("Starting hyperparameter tuning...")
        
        self.initialize_models()
        param_grids = self.get_hyperparameter_grids()
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTuning hyperparameters for {name}...")
            
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.best_models[name] = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Store detailed results
            self.model_scores[name] = {
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'grid_search': grid_search
            }
        
        return self.best_models, self.model_scores
    
    def evaluate_models(self, X_test, y_test, use_best_models=True):
        """Evaluate models on test set"""
        print("Evaluating models on test set...")
        
        models_to_evaluate = self.best_models if use_best_models and self.best_models else self.models
        evaluation_results = {}
        
        for name, model in models_to_evaluate.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"Precision (Class 1): {class_report['1']['precision']:.4f}")
            print(f"Recall (Class 1): {class_report['1']['recall']:.4f}")
            print(f"F1-score (Class 1): {class_report['1']['f1-score']:.4f}")
        
        return evaluation_results
    
    def plot_model_comparison(self, evaluation_results, save_path=None):
        """Plot comparison of model performance"""
        model_names = list(evaluation_results.keys())
        metrics = ['accuracy', 'roc_auc']
        
        # Extract precision, recall, f1 for class 1 (churn)
        precisions = [evaluation_results[name]['classification_report']['1']['precision'] for name in model_names]
        recalls = [evaluation_results[name]['classification_report']['1']['recall'] for name in model_names]
        f1_scores = [evaluation_results[name]['classification_report']['1']['f1-score'] for name in model_names]
        accuracies = [evaluation_results[name]['accuracy'] for name in model_names]
        roc_aucs = [evaluation_results[name]['roc_auc'] for name in model_names]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # ROC-AUC comparison
        axes[0, 0].bar(model_names, roc_aucs, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('ROC-AUC Score')
        axes[0, 0].set_ylabel('ROC-AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # Accuracy comparison
        axes[0, 1].bar(model_names, accuracies, color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_ylim(0, 1)
        
        # Precision, Recall, F1 comparison
        x_pos = np.arange(len(model_names))
        width = 0.25
        
        axes[1, 0].bar(x_pos - width, precisions, width, label='Precision', alpha=0.8)
        axes[1, 0].bar(x_pos, recalls, width, label='Recall', alpha=0.8)
        axes[1, 0].bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[1, 0].set_title('Precision, Recall, F1-Score (Churn Class)')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(model_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].set_ylim(0, 1)
        
        # Model ranking based on ROC-AUC
        model_ranking = sorted(zip(model_names, roc_aucs), key=lambda x: x[1], reverse=True)
        ranks = [i+1 for i in range(len(model_names))]
        ranked_names = [name for name, _ in model_ranking]
        ranked_scores = [score for _, score in model_ranking]
        
        axes[1, 1].barh(ranks, ranked_scores, color='orange', alpha=0.8)
        axes[1, 1].set_title('Model Ranking (ROC-AUC)')
        axes[1, 1].set_xlabel('ROC-AUC Score')
        axes[1, 1].set_yticks(ranks)
        axes[1, 1].set_yticklabels(ranked_names)
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, evaluation_results, save_path=None):
        """Plot confusion matrices for all models"""
        n_models = len(evaluation_results)
        cols = min(n_models, 3)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (name, results) in enumerate(evaluation_results.items()):
            cm = results['confusion_matrix']
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                ax=axes[idx],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn']
            )
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices plot saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, X_test, y_test, evaluation_results, save_path=None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, results in evaluation_results.items():
            y_pred_proba = results['prediction_probabilities']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves plot saved to {save_path}")
        
        return plt.gcf()
    
    def get_feature_importance(self, feature_names, model_name=None):
        """Get feature importance from the best model"""
        if model_name:
            if model_name in self.best_models:
                model = self.best_models[model_name]
            elif model_name in self.models:
                model = self.models[model_name]
            else:
                raise ValueError(f"Model {model_name} not found")
        else:
            # Use the best performing model based on ROC-AUC
            best_model_name = max(self.model_scores.keys(), 
                                key=lambda x: self.model_scores[x].get('best_score', 
                                                                   self.model_scores[x].get('cv_mean', 0)))
            model = self.best_models.get(best_model_name, self.models[best_model_name])
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model doesn't have feature importance attribute")
            return None
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def plot_feature_importance(self, feature_names, model_name=None, top_n=20, save_path=None):
        """Plot feature importance"""
        feature_importance_df = self.get_feature_importance(feature_names, model_name)
        
        if feature_importance_df is None:
            return None
        
        # Take top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(range(len(top_features)), top_features['importance'][::-1])
        plt.yticks(range(len(top_features)), top_features['feature'][::-1])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf(), feature_importance_df
    
    def save_best_model(self, model_name, filepath):
        """Save the best model to disk"""
        if model_name in self.best_models:
            model = self.best_models[model_name]
        elif model_name in self.models:
            model = self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(model, filepath)
        print(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    
    def get_model_summary(self):
        """Get summary of all trained models"""
        summary = {}
        
        for name in self.models.keys():
            model_info = {
                'base_model_scores': self.model_scores.get(name, {})
            }
            
            if name in self.best_models:
                model_info['best_model'] = self.best_models[name]
                model_info['hyperparameter_tuning'] = True
            else:
                model_info['hyperparameter_tuning'] = False
            
            summary[name] = model_info
        
        return summary
