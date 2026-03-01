import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ==================== ENSEMBLE MODEL FOR IRIS ====================
print("="*60)
print("BUILDING ENSEMBLE MODEL FOR IRIS DATASET")
print("="*60)

iris = pd.read_csv('Datasets/iris_cleaned.csv')
print(f"Dataset shape: {iris.shape}")
print(f"Features: {iris.columns.tolist()}")

X = iris.drop('Species', axis=1)
y = iris['Species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svm', svm)
    ],
    voting='soft'
)

print("\nTraining ensemble model...")
ensemble.fit(X_train_scaled, y_train)

y_pred = ensemble.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nEnsemble Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs('models', exist_ok=True)
joblib.dump(ensemble, 'models/ensemble_model.pkl')
joblib.dump(scaler, 'models/iris_scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
print("\nModels saved to models/")

# ==================== NEURAL NETWORK FOR STUDENT DATA ====================
print("\n" + "="*60)
print("BUILDING NEURAL NETWORK FOR STUDENT DATASET")
print("="*60)

std = pd.read_csv('Datasets/std_cleaned.csv')
print(f"Dataset shape: {std.shape}")
print(f"Features: {std.columns.tolist()}")

X_std = std.drop('GPA', axis=1)
y_std = std['GPA']

X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y_std, test_size=0.2, random_state=42)

scaler_std = StandardScaler()
X_train_std_scaled = scaler_std.fit_transform(X_train_std)
X_test_std_scaled = scaler_std.transform(X_test_std)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_std.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("\nTraining Neural Network...")
history = model.fit(
    X_train_std_scaled, y_train_std,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

loss, mae = model.evaluate(X_test_std_scaled, y_test_std)
print(f"\nNeural Network - Test Loss: {loss:.4f}")
print(f"Neural Network - Test MAE: {mae:.4f}")

model.save('models/neural_network_model.keras')
joblib.dump(scaler_std, 'models/std_scaler.pkl')
print("\nNeural Network model saved!")
