"""
Data preprocessing utilities for Customer Churn Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os

# Note: imbalanced-learn is optional - we can work without it

class ChurnDataPreprocessor:
    """Preprocessor for Customer Churn data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        self.categorical_features = []
        self.target_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        """Load the dataset from CSV file"""
        self.df = pd.read_csv(file_path)
        return self.df
    
    def basic_cleaning(self, df):
        """Perform basic data cleaning"""
        df = df.copy()
        
        # Handle TotalCharges column - convert to numeric
        if 'TotalCharges' in df.columns:
            # Replace empty strings with NaN
            df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Fill missing TotalCharges with 0 (new customers)
            df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Remove customerID if present (not useful for prediction)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            
        return df
    
    def feature_engineering(self, df):
        """Create new features from existing ones"""
        df = df.copy()
        
        # Calculate average monthly charges
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['AvgMonthlyCharges'] = np.where(
                df['tenure'] > 0, 
                df['TotalCharges'] / df['tenure'], 
                df['MonthlyCharges']
            )
        
        # Create tenure groups
        if 'tenure' in df.columns:
            df['TenureGroup'] = pd.cut(
                df['tenure'], 
                bins=[0, 12, 24, 48, 72], 
                labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
            )
        
        # Create charge level categories
        if 'MonthlyCharges' in df.columns:
            df['ChargeLevel'] = pd.cut(
                df['MonthlyCharges'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Service count (number of services customer has)
        service_cols = [
            'PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        service_count = 0
        for col in service_cols:
            if col in df.columns:
                service_count += (df[col] == 'Yes').astype(int)
        
        df['ServiceCount'] = service_count
        
        return df
    
    def identify_feature_types(self, df, target_col='Churn'):
        """Identify numeric and categorical features"""
        # Exclude target column
        features = [col for col in df.columns if col != target_col]
        
        self.numeric_features = []
        self.categorical_features = []
        
        for col in features:
            if df[col].dtype in ['int64', 'float64']:
                self.numeric_features.append(col)
            else:
                self.categorical_features.append(col)
        
        print(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        return self.numeric_features, self.categorical_features
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using label encoding"""
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns:
                if fit:
                    # Fit and transform
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    # Transform only
                    if col in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(df_encoded[col].astype(str))
                        known_values = set(self.label_encoders[col].classes_)
                        
                        if not unique_values.issubset(known_values):
                            # Add unseen categories to the encoder
                            new_values = list(unique_values - known_values)
                            self.label_encoders[col].classes_ = np.append(
                                self.label_encoders[col].classes_, new_values
                            )
                        
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def scale_numeric_features(self, df, fit=True):
        """Scale numeric features"""
        df_scaled = df.copy()
        
        if self.numeric_features:
            if fit:
                df_scaled[self.numeric_features] = self.scaler.fit_transform(df_scaled[self.numeric_features])
            else:
                df_scaled[self.numeric_features] = self.scaler.transform(df_scaled[self.numeric_features])
        
        return df_scaled
    
    def prepare_features_target(self, df, target_col='Churn'):
        """Separate features and target variable"""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Encode target variable
        if y.dtype == 'object':
            y = self.target_encoder.fit_transform(y)
        
        return X, y
    
    def full_preprocessing_pipeline(self, df, target_col='Churn', test_size=0.2, random_state=42):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing pipeline...")
        
        # Basic cleaning
        df_clean = self.basic_cleaning(df)
        print(f"After cleaning: {df_clean.shape}")
        
        # Feature engineering
        df_engineered = self.feature_engineering(df_clean)
        print(f"After feature engineering: {df_engineered.shape}")
        
        # Identify feature types
        self.identify_feature_types(df_engineered, target_col)
        
        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_engineered, fit=True)
        print("Categorical features encoded")
        
        # Scale numeric features
        df_scaled = self.scale_numeric_features(df_encoded, fit=True)
        print("Numeric features scaled")
        
        # Prepare features and target
        X, y = self.prepare_features_target(df_scaled, target_col)
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Target distribution in train set: {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, df):
        """Transform new data using fitted preprocessors"""
        # Basic cleaning
        df_clean = self.basic_cleaning(df)
        
        # Feature engineering
        df_engineered = self.feature_engineering(df_clean)
        
        # Encode categorical features (transform only)
        df_encoded = self.encode_categorical_features(df_engineered, fit=False)
        
        # Scale numeric features (transform only)
        df_scaled = self.scale_numeric_features(df_encoded, fit=False)
        
        return df_scaled
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor for later use"""
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath):
        """Load a saved preprocessor"""
        preprocessor_data = joblib.load(filepath)
        
        self.label_encoders = preprocessor_data['label_encoders']
        self.scaler = preprocessor_data['scaler']
        self.target_encoder = preprocessor_data['target_encoder']
        self.numeric_features = preprocessor_data['numeric_features']
        self.categorical_features = preprocessor_data['categorical_features']
        
        print(f"Preprocessor loaded from {filepath}")

def get_feature_descriptions():
    """Get descriptions of features in the dataset"""
    return {
        'gender': 'Customer gender (Male/Female)',
        'SeniorCitizen': 'Whether customer is senior citizen (0/1)',
        'Partner': 'Whether customer has partner (Yes/No)',
        'Dependents': 'Whether customer has dependents (Yes/No)',
        'tenure': 'Number of months customer has stayed',
        'PhoneService': 'Whether customer has phone service (Yes/No)',
        'MultipleLines': 'Whether customer has multiple lines',
        'InternetService': 'Type of internet service (DSL/Fiber optic/No)',
        'OnlineSecurity': 'Whether customer has online security (Yes/No/No internet service)',
        'OnlineBackup': 'Whether customer has online backup (Yes/No/No internet service)',
        'DeviceProtection': 'Whether customer has device protection (Yes/No/No internet service)',
        'TechSupport': 'Whether customer has tech support (Yes/No/No internet service)',
        'StreamingTV': 'Whether customer has streaming TV (Yes/No/No internet service)',
        'StreamingMovies': 'Whether customer has streaming movies (Yes/No/No internet service)',
        'Contract': 'Contract term (Month-to-month/One year/Two year)',
        'PaperlessBilling': 'Whether customer has paperless billing (Yes/No)',
        'PaymentMethod': 'Payment method used by customer',
        'MonthlyCharges': 'Monthly charges amount',
        'TotalCharges': 'Total amount charged to customer',
        'Churn': 'Whether customer churned (Yes/No)'
    }
