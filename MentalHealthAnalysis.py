#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# class TrainDataCleaner:
#     def __init__(self, train):
#         """Initialize with a dataset"""
#         self.train = train.copy()

#     def clean_data(self):
#         """Apply all data cleaning steps efficiently"""
#         self.train.loc[self.train['Working Professional or Student'] == 'Student', 'Profession'] = \
#             self.train.loc[self.train['Working Professional or Student'] == 'Student', 'Profession'].fillna('Student')
#         self.train.dropna(subset=['Profession'], inplace=True)

#         for col, role in [('Academic Pressure', 'Working Professional'), ('Work Pressure', 'Student')]:
#             self.train.loc[self.train['Working Professional or Student'] == role, col] = \
#                 self.train.loc[self.train['Working Professional or Student'] == role, col].fillna(0)
#             self.train.dropna(subset=[col], inplace=True)

#         for col in ['CGPA', 'Study Satisfaction']:
#             self.train.loc[self.train['Working Professional or Student'] == 'Working Professional', col] = \
#                 self.train.loc[self.train['Working Professional or Student'] == 'Working Professional', col].fillna(0)
#             self.train.dropna(subset=[col], inplace=True)

#         self.train.loc[self.train['Working Professional or Student'] == 'Student', 'Job Satisfaction'] = \
#             self.train.loc[self.train['Working Professional or Student'] == 'Student', 'Job Satisfaction'].fillna(0)
#         self.train.dropna(subset=['Job Satisfaction'], inplace=True)

#         return self.train


class TestDataCleaner:
    def __init__(self, test):
        """Initialize with a dataset"""
        self.test = test.copy()

    def clean_data(self):
        """Apply all data cleaning steps efficiently"""
        self.test.loc[self.test['Working Professional or Student'] == 'Student', 'Profession'] = \
            self.test.loc[self.test['Working Professional or Student'] == 'Student', 'Profession'].fillna('Student')

        for col, role in [('Academic Pressure', 'Working Professional'), ('Work Pressure', 'Student')]:
            self.test.loc[self.test['Working Professional or Student'] == role, col] = \
                self.test.loc[self.test['Working Professional or Student'] == role, col].fillna(0)

        for col in ['CGPA', 'Study Satisfaction']:
            self.test.loc[self.test['Working Professional or Student'] == 'Working Professional', col] = \
                self.test.loc[self.test['Working Professional or Student'] == 'Working Professional', col].fillna(0)

        self.test.loc[self.test['Working Professional or Student'] == 'Student', 'Job Satisfaction'] = \
            self.test.loc[self.test['Working Professional or Student'] == 'Student', 'Job Satisfaction'].fillna(0)

        return self.test


# def encode_categorical_data(train, test, categorical_cols, encoder_file="encoder.pkl"):
#     """Encodes categorical data and saves encoders"""
    
#     encoders = {}  # Store encoders for reuse
#     one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

#     # Fit the encoder on the training data
#     one_hot_encoder.fit(train[categorical_cols])

#     # Transform the training and test data
#     train_encoded = pd.DataFrame(one_hot_encoder.transform(train[categorical_cols]))
#     test_encoded = pd.DataFrame(one_hot_encoder.transform(test[categorical_cols]))

#     # Update column names based on the categories in categorical columns
#     train_encoded.columns = one_hot_encoder.get_feature_names_out(categorical_cols)
#     test_encoded.columns = one_hot_encoder.get_feature_names_out(categorical_cols)

#     encoders["one_hot"] = one_hot_encoder  # Save the one-hot encoder

#     # Save the encoder for future use
#     with open(encoder_file, "wb") as f:
#         pickle.dump(encoders, f)

#     return train_encoded, test_encoded

# **Only execute the following if the script is run directly**
# if __name__ == "__main__":
#     # Loading Dataset
#     train = pd.read_csv("train.csv")
#     test = pd.read_csv("test.csv")

#     # Cleaning Data
#     cleaner = TrainDataCleaner(train)
#     train_cleaned = cleaner.clean_data()

#     cleaner = TestDataCleaner(test)
#     test_cleaned = cleaner.clean_data()

#     # Encoding Data
#     categorical_cols = ['Gender', 'City', 'Profession', 'Degree', 'Dietary Habits', 
#                         'Sleep Duration', 'Family History of Mental Illness', 'Working Professional or Student', 
#                         'Have you ever had suicidal thoughts ?']

#     train_encoded, train_y, test_encoded = encode_categorical_data(
#         train=train_cleaned, test=test_cleaned, categorical_cols=categorical_cols, target_col='Depression'
#     )

#     print("Data processing completed successfully!")
