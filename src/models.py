from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import json
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42)
        }

        self.param_grids = {
            'decision_tree': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 7, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }

        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}

    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        print(f'Training {model_name}...')

        # Perform grid Search
        grid_search = GridSearchCV(
            self.models[model_name],
            self.param_grids[model_name],
            cv=5,
            scoring='accuracy'
        )

        grid_search.fit(X_train, y_train)

        # Save best model and parameters
        self.best_models[model_name] = grid_search.best_estimator_
        self.best_params[model_name] = grid_search.best_params_

        # Evulate on validation set
        y_pred = grid_search.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_pred)
        self.best_scores[model_name]= {
            'validation_accuracy': val_accuracy,
            'best_cv_score': grid_search.best_score_,
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }

        print(f'Best validation accuracy for {model_name}: {val_accuracy:.4f}')
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train, X_val, y_val)

    def save_results(self, output_dir='../data/models'):
        # Create directory if it doesn't exist:
        os.makedirs(output_dir, exist_ok=True)

        # Save models
        for name, model in self.best_models.items():
            joblib.dump(model, f'{output_dir}/{name}_best.joblib')

        #Save parameters and scores
        results = {
            'best_params': self.best_params,
            'best_scores': self.best_scores
        }

        with open(f'{output_dir}/training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
