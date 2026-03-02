# 🎓 Intelligent Systems Project

## 📋 Project Overview

This project contains two machine learning models:
1. **Ensemble Model** - Iris flower classification (Random Forest + Gradient Boosting + SVM)
2. **Neural Network** - Student GPA prediction

## 📁 Project Structure

```
Intell/
├── Datasets/
│   ├── iris_dirty.csv       # Original dirty Iris data
│   ├── iris_cleaned.csv      # Cleaned Iris data
│   ├── std.csv              # Original student data
│   └── std_cleaned.csv      # Cleaned student data
├── models/
│   ├── ensemble_model.pkl   # Trained ensemble model
│   ├── iris_scaler.pkl      # Iris data scaler
│   ├── label_encoder.pkl    # Species encoder
│   ├── neural_network_model.keras  # Neural network model
│   └── std_scaler.pkl       # Student data scaler
├── app.py                   # Streamlit web application
├── train_models.py          # Model training script
├── data_preparation.py      # Data cleaning script
└── requirements.txt         # Python dependencies
```

## 🚀 Deployment Instructions

### Option 1: Streamlit Community Cloud (Recommended)

1. **Create GitHub Repository**
   - Go to https://github.com/new
   - Repository name: `intell-project`
   - Make it **Public**
   - Click "Create repository"

2. **Push Code to GitHub**
   ```bash
   cd Intell
   gh auth login
   gh repo create intell-project --public --source=. --push
   ```
   Or use GitHub Desktop / Git

3. **Deploy to Streamlit**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `orasa205/intell-project`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

### Option 2: Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run data preparation
python data_preparation.py

# Train models
python train_models.py

# Run web app
streamlit run app.py
```

## 📊 Model Performance

| Model | Dataset | Performance |
|-------|---------|-------------|
| Ensemble (RF+GB+SVM) | Iris | 96.67% accuracy |
| Neural Network (64-32-16) | Student | 0.17 MAE |

## 📝 Requirements

- Python 3.8+
- Streamlit
- TensorFlow
- Scikit-learn
- Pandas
- NumPy

## 👨‍🎓 Project Information

- **Course**: Intelligent Systems
- **Student**: Orasa Sapram
- **Year**: 2026
