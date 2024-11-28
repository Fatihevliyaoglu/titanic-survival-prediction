from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson']
        self.categorical_features = ['Sex', 'Embarked', 'Title', 'AgeBin', 
                                   'FamilyType', 'FareBin']
        self.binary_features = ['IsChild', 'IsAlone', 'HasCabin']
        
    def fit(self, X, y=None):
        self.preprocessor = self._create_preprocessor()
        self.preprocessor.fit(X)
        return self
        
    def transform(self, X):
        return self.preprocessor.transform(X)
    
    def _create_preprocessor(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('bin', binary_transformer, self.binary_features)
            ])
        
        return preprocessor