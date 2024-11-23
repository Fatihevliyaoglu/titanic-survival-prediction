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
│   ├── 01_EDA.ipynb                     # Exploratory analysis
│   ├── 02_Feature_Engineering.ipynb     # Feature creation
│   ├── 03_Model_Development.ipynb       # Individual models
│   ├── 04_Ensemble_Building.ipynb       # Ensemble creation
│   └── 05_Final_Evaluation.ipynb        # Final evaluation
│
├── src/
│   ├── preprocessing.py   # Data preprocessing
│   ├── models.py          # Base models
│   ├── tuning.py          # Hyperparameter optimization
│   └── ensemble.py        # Ensemble methods
│
├── requirements.txt
├── README.md
└── .gitignore
``` 

## Features

- Decision Tree classifier hyperparameter tuning
- Random Forest optimization
- XGBoost implementation
- Stacking ensemble of optimized models
- Cross-validation strategy
- Feature importance analysis

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

1. EDA for data understanding
2. Feature engineering
3. Base model development
4. Ensemble creation
5. Final model evaluation

## Model Architecture

- Base Models:
  - Decision Tree
  - Random Forest
  - XGBoost
- Hyperparameter Optimization: GridSearchCV
- Final Ensemble: Stacking with LogisticRegression meta-learner

## Results

- Current Kaggle score: [Your Score]
- Best performing model: [Model Name]
- Key insights:
  - [Insight 1]
  - [Insight 2]
  - [Insight 3]

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT

## Author

Fatih Evliyaoglu

GitHub: @fatihevliyaoglu

LinkedIn: [https://www.linkedin.com/in/fatih-evliyaoglu-aa5a6019b/]