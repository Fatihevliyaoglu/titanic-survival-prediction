import numpy as np 
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score, classification_report
import json
import os

class EnsembleBuilder:
    def __init__(self, base_models, meta_learner=None):
        """
        Initialize ensemble builder

        Parameters:
        base_models (dict)_ Dictionary of trained models
        meta_learner: Meta classifier (defaults to LogisticRegression)
        """

        self.base_models = base_models
        self.meta_learner = meta_learner or LogisticRegression(random_state=42)
        self.ensemble = None

    def build_stacking(self):
        """Create stacking ensemble"""
        estimators = [
            (name, model) for name, model in self.base_models.items()
        ]

        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            cv=5,
            passthrough=True
        )

        return self.ensemble
    
    def fit_ensemble(self, X_train, y_train):
        """Fit the ensemble model"""
        if self.ensemble is None:
            self.build_stacking()

        self.ensemble.fit(X_train, y_train)

        return self
    
    def evaluate(self, X_val, y_val):
        """Evaluate ensemble performence"""
        y_pred = self.ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)

        results = {
            'accuracy': accuracy,
            'classification_report': report
        }

        return results
    
    def save_ensemble(self, output_dir='../data/models'):
        """Save ensemble model and results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save ensemble model
        joblib.dump(self.ensemble, f'{output_dir}/stacking_ensemble.joblib')