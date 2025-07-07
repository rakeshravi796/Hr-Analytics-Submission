

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle

# Mapping for sanitizing job titles
SANITIZE_JOB_TITLE_DICT = {
    "Data Analyst": ["Data Analyst"],
    "DevOps Engineer": ["DevOps Engineer"],
    "Research Scientist": ["Research Scientist"],
    "Software Engineer": ["Sofware Engneer", "Software Engr", "Softwre Engineer", "Software Engineer"],
    "Data Scientist": ["Dt Scientist", "Data Scienist", "Data Scntist", "Data Scientist"],
    "ML Engineer": ["ML Engr", "ML Enginer", "Machine Learning Engr", "ML Engineer"]
}

COLUMNS_TO_KEEP = [
    'job_title',
    'experience_level',
    'employment_type',
    'company_size',
    'company_location',
    'remote_ratio',
    'salary_currency',
    'years_experience',
    'salary_in_usd'
]

class DataCleaner(BaseEstimator, TransformerMixin):
    def sanitize_job_title(self, s: str) -> str:
        for key, variants in SANITIZE_JOB_TITLE_DICT.items():
            if s in variants:
                return key
        return "Not Determinable"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.drop(columns=['education', 'skills'], errors='ignore')

        if 'job_title' in df.columns:
            df['job_title'] = df['job_title'].apply(self.sanitize_job_title)

        df = df[[col for col in COLUMNS_TO_KEEP if col in df.columns]]
        return df

def build_pipeline():
    return Pipeline([('data_cleaning', DataCleaner())])

def save_pipeline(pipeline, path="Pickle/data_cleaning_pipeline.pkl"):
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
