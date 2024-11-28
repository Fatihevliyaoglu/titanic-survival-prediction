# Titanic Survival Prediction

## Overwiev
Advanced machine learning project predicting Titanic passenger survival using ensemble methods and hyperparameter optimization.

## Project Structure

```
titanic-survival-prediction/
│
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Processed datasets
│   ├── models/             # Saved model files
│   └── predictions/        # Model predictions
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory analysis
│   ├── 02_Feature_Engineering.ipynb     # Feature creation
│   ├── 03_Preprocessing.ipynb          # Data preprocessing
│   ├── 04_Model_Development.ipynb      # Model training
│   ├── 05_Ensemble_Building.ipynb      # Ensemble creation
│   └── 06_Final_Prediction.ipynb       # Prediction generation
│
├── src/
│   ├── preprocessing.py    # Preprocessing pipeline
│   ├── models.py          # Model implementations
│   └── ensemble.py        # Ensemble methods
│   └── predict.py         # Prediction pipeline
│
├── requirements.txt
├── README.md
└── .gitignore
``` 

# Models Implemented

- Decision Tree classifier
- Random Forest classifier
- XGBoost classifier
- Stacking ensemble

## Features
- Comprehensive data preprocessing
- Feature engineering
- Model hyperparameter tuning
- Ensemble modeling
- Cross-validation strategy

## Installation

```
# Clone the repository
git clone https://github.com/Fatihevliyaoglu/titanic-survival-prediction.git

# Navigate to the project directory
cd titanic-survival-prediction

# Install required packages
pip install -r requirements.txt
```

## Usage

Follow notebooks in numerical order:

- Run notebooks in order:
  - Start with EDA
  - Process through model development
  - End with predictions

- Final submission file will be generated in data/predictions/

## Model Architecture

- Base Models:
  - Decision Tree
  - Random Forest
  - XGBoost
- Hyperparameter Optimization: GridSearchCV
- Final Ensemble: Stacking with LogisticRegression meta-learner

## Results

- Best performing model: DecisionTree: 85,47%
- Key insights:
  - Gender and Class are the strongest predictors of survival (females and higher-class passengers had higher survival rates)
  - Family size impacts survival - passengers traveling in small family groups (2-4 members) had better survival rates than those traveling alone or in large groups
  - Age plays a significant role - children (especially from first and second class) had higher survival rates
  - Cabin location (Deck) shows correlation with survival, though this data is partially missing
  - Fare amount correlates with survival, likely due to its relationship with passenger class

## License
MIT

## Author

Fatih Evliyaoglu

GitHub: @fatihevliyaoglu

LinkedIn: [https://www.linkedin.com/in/fatih-evliyaoglu-aa5a6019b/]
