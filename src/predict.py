import pandas as pd 
import numpy as np 
import joblib 

class PredictionPipeline:
    def __init__(self, model_path='../data/models/stacking_ensemble.joblib'):
        self.model = joblib.load(model_path)

    def predict(self, test_data):
        """Generate predictions"""

        # Make predictions
        predictions = self.model.predict(test_data)

        return predictions
    
    def create_submission(self, test_data, output_path='../data/predictions/submission.csv'):
        """Create submission file"""
        predictions = self.predict(test_data)

        raw_test = pd.read_csv('../data/raw/test.csv')

        submission = pd.DataFrame({
            'PassengerId': raw_test['PassengerId'].astype(int),
            'Survived': predictions.astype(int)
        })

        # Save submission
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission
