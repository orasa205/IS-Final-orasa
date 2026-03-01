import pandas as pd
import numpy as np

# ==================== IRIS DATA PREPARATION ====================
print("="*50)
print("IRIS DATASET PREPARATION")
print("="*50)

iris = pd.read_csv('Datasets/iris_dirty.csv', index_col=0)
print(f"Original shape: {iris.shape}")
print(f"\nMissing values before:\n{iris.isnull().sum()}")
print(f"\nSpecies value counts (before):\n{iris['Species'].value_counts()}")

iris_cleaned = iris.copy()

iris_cleaned['Species'] = iris_cleaned['Species'].str.lower().str.strip()

for col in ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']:
    iris_cleaned[col] = iris_cleaned[col].fillna(iris_cleaned[col].median())

print(f"\nMissing values after:\n{iris_cleaned.isnull().sum()}")
print(f"\nSpecies value counts (after):\n{iris_cleaned['Species'].value_counts()}")

iris_cleaned.to_csv('Datasets/iris_cleaned.csv', index=False)
print("\niris_cleaned.csv saved!")

# ==================== STD DATA PREPARATION ====================
print("\n" + "="*50)
print("STUDENT DATASET PREPARATION")
print("="*50)

std = pd.read_csv('Datasets/std.csv')
print(f"Original shape: {std.shape}")

std_cleaned = std.copy()

for col in ['Reading', 'Notes', 'Listening_in_Class', 'Project_work']:
    std_cleaned[col] = std_cleaned[col].astype(str).str.lower().str.strip()
    std_cleaned[col] = std_cleaned[col].replace({'yes': 1, 'no': 0, '6': 0})

std_cleaned['Sex'] = std_cleaned['Sex'].str.lower().str.strip()
std_cleaned['Additional_Work'] = std_cleaned['Additional_Work'].str.lower().str.strip()
std_cleaned['Sports_activity'] = std_cleaned['Sports_activity'].str.lower().str.strip()
std_cleaned['Transportation'] = std_cleaned['Transportation'].str.lower().str.strip()

std_cleaned['Sex'] = std_cleaned['Sex'].map({'male': 0, 'female': 1})
std_cleaned['Additional_Work'] = std_cleaned['Additional_Work'].map({'yes': 1, 'no': 0})
std_cleaned['Sports_activity'] = std_cleaned['Sports_activity'].map({'yes': 1, 'no': 0})
std_cleaned['Transportation'] = std_cleaned['Transportation'].map({'private': 1, 'bus': 0})

print(f"\nCleaned std dataset:")
print(std_cleaned.head())

std_cleaned.to_csv('Datasets/std_cleaned.csv', index=False)
print("\nstd_cleaned.csv saved!")
