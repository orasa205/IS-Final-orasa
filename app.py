import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Intell - Intelligent Systems Project",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        margin-top: 30px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #e74c3c;
        margin-top: 20px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .code-block {
        background-color: #2d2d2d;
        color: #f8f8f2;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .reference-link {
        color: #3498db;
        text-decoration: none;
    }
    .reference-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📚 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "📊 Student Performance(NN)",
    "🤖 Ensemble Model (Machine Learning)",
    "🧪 Test Ensemble Model",
    "🧠 Neural Network Model",
    "🧪 Test Neural Network",
    "🔥 Train Iris Neural Network"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Project Info")
st.sidebar.info("""
**Student:** Orasa Sapram

**Year:** 2026

**Course:** Intelligent Systems
""")

if page == "🏠 Home":
    st.markdown('<p class="main-header">🎓 Intelligent Systems Project</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Ensemble Model</h3>
            <h2>96.67%</h2>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 📊 Iris Dataset")
        st.markdown("Iris flower classification (3 species)")
        st.markdown("**Algorithm:** Voting Classifier")
        st.markdown("- Random Forest")
        st.markdown("- Gradient Boosting")
        st.markdown("- SVM")
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Neural Network</h3>
            <h2>0.174</h2>
            <p>Mean Absolute Error (MAE)</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 📊 Student Performance Dataset")
        st.markdown("Student GPA prediction")
        st.markdown("**Architecture:** MLP 3-Layer")
        st.markdown("- Input: 11 features")
        st.markdown("- Hidden: 64-32-16 neurons")
        st.markdown("- Output: 1 neuron (linear)")
    
    st.markdown("---")
    st.markdown("## 📈 Project Overview")
    
    col3, col4 = st.columns(2)
    with col3:
        st.info("""
        ### 📁 Datasets
        - **iris_dirty.csv** - Iris flower data
        - **std.csv** - Student performance data
        """)
    with col4:
        st.info("""
        ### 📦 Models
        - **ensemble_model.pkl** - Ensemble ML
        - **neural_network_model.keras** - Neural Network
        """)
    
    st.markdown("---")
    st.markdown("""
    ## 📋 Project Description
    
    This project demonstrates the development of two intelligent systems:
    
    ### 1. Ensemble Machine Learning Model
    Used for classifying Iris flower species by combining multiple models
    
    ### 2. Neural Network Model  
    Used for predicting student GPA based on various factors
    
    ---
    ### 👨‍🎓 By: Orasa Sapram
    ### 📅 Year: 2026
    """)
    
elif page == "🤖 Ensemble Model (Machine Learning)":
    st.markdown('<p class="main-header">🤖 Ensemble Model (Machine Learning)</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("## 1. 📂 Data Preparation")
    
    st.markdown("""
    <div class="info-box">
    <b>Data Source:</b> Kaggle - Iris Dataset (dirty version)<br>
    <b>Number of Samples:</b> 150<br>
    <b>Number of Features:</b> 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)<br>
    <b>Number of Classes:</b> 3 (Setosa, Versicolor, Virginica)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Issues Found in Raw Data:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Missing Values</b>
        <ul>
            <li>Sepal.Length: 8 missing values</li>
            <li>Sepal.Width: 11 missing values</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Inconsistent Labels</b>
        <ul>
            <li>Setosa, setosa, SETOSA</li>
            <li>Versicolor, versicolor, VERSICOLOR</li>
            <li>Virginica, virginica, VIRGINICA</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Data Cleaning Steps:")
    
    st.code("""
# 1. Fill missing values with median
df['Sepal.Length'].fillna(df['Sepal.Length'].median(), inplace=True)
df['Sepal.Width'].fillna(df['Sepal.Width'].median(), inplace=True)

# 2. Normalize Species to lowercase
df['Species'] = df['Species'].str.lower()

# 3. Save cleaned file
df.to_csv('Datasets/iris_cleaned.csv', index=False)
    """, language="python")
    
    st.markdown("---")
    st.markdown("## 2. 📚 Algorithm Theory")
    
    st.markdown("""
    <div class="info-box">
    <b>Ensemble Learning</b> combines multiple models together to achieve better performance than individual models
    </div>
    """, unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.markdown("### 🌲 Random Forest")
        st.markdown("""
        **Bagging Method**
        
        - Build multiple Decision Trees
        - Each tree uses random data
        - Combine results with Voting
        
        **Advantages:**
        - Reduces overfitting
        - Works well with noisy data
        - Fast processing
        """)
    
    with col4:
        st.markdown("### 🚀 Gradient Boosting")
        st.markdown("""
        **Boosting Method**
        
        - Build models sequentially
        - Learn from previous errors (residuals)
        - Reduces bias and variance
        
        **Advantages:**
        - High accuracy
        - Handles data well
        """)
    
    with col5:
        st.markdown("### 📊 SVM")
        st.markdown("""
        **Support Vector Machine**
        
        - Find Hyperplane to separate data
        - Use RBF Kernel
        - Maximum Margin Classifier
        
        **Advantages:**
        - Works well in high-dimensional space
        - Memory efficient
        """)
    
    st.markdown("### Voting Classifier (Soft Voting)")
    st.markdown("""
    Combines predictions from all models by averaging probabilities (Probability Averaging)
    
    $$P(y|x) = \\frac{1}{K} \\sum_{k=1}^{K} P_k(y|x)$$
    
    Where K is the number of models in the Ensemble
    """)
    
    st.markdown("---")
    st.markdown("## 3. 🔧 Model Development Steps")
    
    steps = [
        ("1. Load Data", "Read iris_cleaned.csv"),
        ("2. Split Data", "Separate X (features) and y (target)"),
        ("3. Encode Labels", "Use LabelEncoder for Species"),
        ("4. Feature Scaling", "Apply StandardScaler"),
        ("5. Train-Test Split", "80% Training, 20% Testing"),
        ("6. Create Ensemble", "Combine RF + GB + SVM"),
        ("7. Train Model", "Fit with Training Data"),
        ("8. Evaluate", "Calculate Accuracy and Report"),
        ("9. Save Model", "Export using joblib")
    ]
    
    for i, (step, desc) in enumerate(steps, 1):
        st.markdown(f"**{step}:** {desc}")
    
    st.markdown("### Model Development Code:")
    st.code("""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Ensemble
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
    voting='soft'
)

# Train
ensemble.fit(X_train_scaled, y_train)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
    """, language="python")
    
    st.markdown("---")
    st.markdown("## 4. 📊 Model Performance")
    
    col6, col7 = st.columns(2)
    with col6:
        st.markdown("""
        <div class="success-box">
        <h3>🎯 Overall Accuracy</h3>
        <h1>96.67%</h1>
        </div>
        """, unsafe_allow_html=True)
    with col7:
        st.markdown("### Classification Report")
        st.markdown("""
        | Class | Precision | Recall | F1-Score |
        |-------|-----------|--------|----------|
        | Setosa | 1.00 | 1.00 | 1.00 |
        | Versicolor | 1.00 | 0.90 | 0.95 |
        | Virginica | 0.91 | 1.00 | 0.95 |
        | **Overall** | **0.97** | **0.97** | **0.97** |
        """)
    
    st.markdown("---")
    st.markdown("## 5. 📖 References")
    
    st.markdown("""
    1. **Iris Dataset**: UCI Machine Learning Repository
       - https://archive.ics.uci.edu/ml/datasets/iris
    
    2. **Scikit-learn Documentation**
       - https://scikit-learn.org/stable/user_guide.html
    
    3. **Ensemble Methods**
       - Breiman, L. (1996). "Bagging Predictors"
       - Breiman, L. (2001). "Random Forests"
    
    4. **Gradient Boosting**
       - Friedman, J. H. (2001). "Greedy Function Approximation"
    
    5. **SVM Theory**
       - Cortes, C., & Vapnik, V. (1995). "Support-vector networks"
    """)
    
elif page == "🧪 Test Ensemble Model":
    st.markdown('<p class="main-header">🧪 Test Ensemble Model</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
    <b>📝 Instructions:</b> Enter Iris flower measurements to predict the species
    </div>
    """, unsafe_allow_html=True)
    
    try:
        model = joblib.load('models/ensemble_model.pkl')
        scaler = joblib.load('models/iris_scaler.pkl')
        le = joblib.load('models/label_encoder.pkl')
        
        st.markdown("### 🌸 Enter Flower Measurements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📏 Sepal")
            sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
            sepal_width = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
            
        with col2:
            st.markdown("#### 🌺 Petal")
            petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
            petal_width = st.number_input("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)
            
        with col3:
            st.markdown("#### 📊 Normal Range")
            st.info("""
            **Sepal Length:** 4.3 - 7.9 cm
            
            **Sepal Width:** 2.0 - 4.4 cm
            
            **Petal Length:** 1.0 - 6.9 cm
            
            **Petal Width:** 0.1 - 2.5 cm
            """)
        
        st.markdown("---")
        
        if st.button("🌺 Predict Species", type="primary"):
            with st.spinner("Predicting..."):
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)
                proba = model.predict_proba(features_scaled)[0]
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            result = le.inverse_transform(prediction)[0].upper()
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if result == "SETOSA":
                    emoji = "🌸"
                    color = "#FF69B4"
                elif result == "VERSICOLOR":
                    emoji = "🌺"
                    color = "#9370DB"
                else:
                    emoji = "🌷"
                    color = "#FFD700"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
                    <h1 style="font-size: 4rem;">{emoji}</h1>
                    <h2 style="color: {color};">{result}</h2>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("#### 📈 Probabilities:")
                for cls, prob in zip(le.classes_, proba):
                    st.markdown(f"**{cls.capitalize()}**: {prob:.1%}")
                    st.progress(prob)
            
            st.markdown("---")
            st.markdown("#### 📝 Description:")
            if result == "SETOSA":
                st.markdown("**Setosa** - Small petals with wide sepals, commonly found in coastal areas")
            elif result == "VERSICOLOR":
                st.markdown("**Versicolor** - Medium-sized flowers, commonly found in North America")
            else:
                st.markdown("**Virginica** - Largest flower species, commonly found in the Americas")
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        
elif page == "🧠 Neural Network Model":
    st.markdown('<p class="main-header">🧠 Neural Network Model</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("## 1. 📂 Data Preparation")
    
    st.markdown("""
    <div class="info-box">
    <b>Data Source:</b> Kaggle - Student Performance Dataset<br>
    <b>Number of Samples:</b> 395<br>
    <b>Number of Features:</b> 11<br>
    <b>Target Variable:</b> GPA (0-4.0)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Issues Found in Raw Data:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Inconsistent Yes/No Values</b>
        <ul>
            <li>yes, Yes, YES</li>
            <li>No, no, NO</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="warning-box">
        <b>⚠️ Outliers</b>
        <ul>
            <li>Listening_in_Class has outliers</li>
            <li>Some values exceed 100</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Feature Details:")
    
    st.markdown("""
    | Feature | Type | Description |
    |---------|------|-------------|
    | age | Numeric | Student age |
    | Sex | Binary | Gender (0=Male, 1=Female) |
    | additonal_work | Binary | Has additional work (0=No, 1=Yes) |
    | sports | Binary | Plays sports (0=No, 1=Yes) |
    | transportation | Binary | Transport (0=Bus, 1=Private) |
    | study_hours | Numeric | Weekly study hours |
    | reading | Binary | Reads (0=No, 1=Yes) |
    | notes | Binary | Takes notes (0=No, 1=Yes) |
    | listening | Numeric | Listening in class (0-100) |
    | project | Binary | Does projects (0=No, 1=Yes) |
    | attendance | Numeric | Attendance (0-100%) |
    | GPA | Numeric | Grade Point Average (0-4.0) - Target |
    """)
    
    st.markdown("### Data Cleaning Steps:")
    st.code("""
# 1. Convert Yes/No to 1/0
binary_cols = ['additonal_work', 'sports', 'reading', 'notes', 'project']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'Yes': 1, 'YES': 1, 'no': 0, 'No': 0, 'NO': 0})

# 2. Convert categorical to numeric
df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
df['transportation'] = df['transportation'].map({'Bus': 0, 'Private': 1})

# 3. Handle outliers
df['listening'] = df['listening'].clip(0, 100)

# 4. Save cleaned file
df.to_csv('Datasets/std_cleaned.csv', index=False)
    """, language="python")
    
    st.markdown("---")
    st.markdown("## 2. 📚 Algorithm Theory")
    
    st.markdown("""
    <div class="info-box">
    <b>Neural Network (Multi-Layer Perceptron - MLP)</b><br>
    A Deep Learning model that simulates the working of neurons in the human brain
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Architecture:")
    
    col2, col3 = st.columns(2)
    with col2:
        st.markdown("""
        **Input Layer (11 neurons)**
        - Accepts 11 features
        
        **Hidden Layer 1 (64 neurons)**
        - Activation: ReLU
        - Dropout: 20%
        - Prevents overfitting
        
        **Hidden Layer 2 (32 neurons)**
        - Activation: ReLU  
        - Dropout: 20%
        
        **Hidden Layer 3 (16 neurons)**
        - Activation: ReLU
        
        **Output Layer (1 neuron)**
        - Activation: Linear
        - Output: GPA (0-4.0)
        """)
    with col3:
        st.markdown("### Activation Functions:")
        st.markdown("""
        **ReLU (Rectified Linear Unit):**
        $$f(x) = max(0, x)$$
        
        **Linear (Output):**
        $$f(x) = x$$
        
        **Optimizer: Adam**
        - Learning Rate: 0.001
        - Adaptive learning rate
        
        **Loss Function: MSE**
        $$MSE = \\frac{1}{n} \\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$
        """)
    
    st.markdown("### Dropout Regularization:")
    st.markdown("""
    Dropout randomly disables some neurons during training to:
    1. Prevent overfitting
    2. Make the model robust to noise
    3. Force neurons to work together
    """)
    
    st.markdown("---")
    st.markdown("## 3. 🔧 Model Development Steps")
    
    steps = [
        ("1. Load Data", "Read std_cleaned.csv"),
        ("2. Split Data", "Separate X (features) and y (GPA)"),
        ("3. Feature Scaling", "Apply StandardScaler"),
        ("4. Train-Test Split", "80% Training, 20% Testing"),
        ("5. Build Architecture", "Use TensorFlow/Keras Sequential"),
        ("6. Compile", "Define Optimizer and Loss"),
        ("7. Train Model", "Fit with Early Stopping"),
        ("8. Evaluate", "Calculate MSE and MAE"),
        ("9. Save Model", "Export as .keras")
    ]
    
    for i, (step, desc) in enumerate(steps, 1):
        st.markdown(f"**{step}:** {desc}")
    
    st.markdown("### Model Development Code:")
    st.code("""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(11,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.4f}")
    """, language="python")
    
    st.markdown("---")
    st.markdown("## 4. 📊 Model Performance")
    
    col4, col5 = st.columns(2)
    with col4:
        st.markdown("""
        <div class="success-box">
        <h3>📉 Test Loss (MSE)</h3>
        <h1>0.048</h1>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown("""
        <div class="success-box">
        <h3>📊 Test MAE</h3>
        <h1>0.174</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    **Interpretation:**
    - MAE = 0.174 means the model predicts GPA with an average error of ~0.17 points
    - This is fairly accurate for predicting academic performance
    """)
    
    st.markdown("---")
    st.markdown("## 5. 📖 References")
    
    st.markdown("""
    1. **Student Performance Dataset**: Kaggle
       - https://www.kaggle.com/datasets
    
    2. **TensorFlow/Keras Documentation**
       - https://www.tensorflow.org/guide/keras/sequential_model
       - https://keras.io/
    
    3. **Deep Learning Theory**
       - Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning"
       - MIT Press
    
    4. **Activation Functions**
       - Nair, V., & Hinton, G. E. (2010). "Rectified linear units improve restricted boltzmann machines"
    
    5. **Adam Optimizer**
       - Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization"
    
    6. **Dropout Regularization**
       - Srivastava, N., et al. (2014). "Dropout: A simple way to prevent neural networks from overfitting"
    """)

elif page == "🧪 Test Neural Network":
    st.markdown('<p class="main-header">🧪 Test Neural Network</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
    <b>📝 Instructions:</b> Enter student information to predict GPA
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from tensorflow.keras.models import load_model
        
        model = load_model('models/neural_network_model.keras')
        scaler = joblib.load('models/std_scaler.pkl')
        
        st.markdown("### 📝 Enter Student Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 General Info")
            age = st.number_input("Age", 15, 25, 18)
            sex = st.selectbox("Gender", ["Male", "Female"])
            transport = st.selectbox("Transportation", ["Bus", "Private"])
            
            st.markdown("#### 📚 Academic Activities")
            study_hours = st.slider("Weekly Study Hours", 0, 30, 10)
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            listening = st.select_slider("Listening in Class", options=["Low", "Medium", "High"], value="Medium")
            
        with col2:
            st.markdown("#### ⭐ Other Activities")
            additional_work = st.selectbox("Additional Work", ["No", "Yes"])
            sports = st.selectbox("Sports", ["No", "Yes"])
            reading = st.selectbox("Reading", ["No", "Yes"])
            notes = st.selectbox("Taking Notes", ["No", "Yes"])
            project = st.selectbox("Projects", ["No", "Yes"])
        
        st.markdown("---")
        
        if st.button("📈 Predict GPA", type="primary"):
            with st.spinner("Predicting..."):
                sex_val = 0 if sex == "Male" else 1
                additional_val = 1 if additional_work == "Yes" else 0
                sports_val = 1 if sports == "Yes" else 0
                transport_val = 1 if transport == "Private" else 0
                reading_val = 1 if reading == "Yes" else 0
                notes_val = 1 if notes == "Yes" else 0
                project_val = 1 if project == "Yes" else 0
                listening_val = 0 if listening == "Low" else (1 if listening == "Medium" else 2)
                
                features = np.array([[age, sex_val, additional_val, sports_val, transport_val,
                                     study_hours, reading_val, notes_val, listening_val, 
                                     project_val, attendance]])
                features_scaled = scaler.transform(features)
                
                prediction = model.predict(features_scaled)[0][0]
                prediction = max(0, min(4, prediction))
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction >= 3.5:
                    emoji = "🌟"
                    grade = "Excellent"
                    color = "#FFD700"
                elif prediction >= 3.0:
                    emoji = "👍"
                    grade = "Good"
                    color = "#28a745"
                elif prediction >= 2.0:
                    emoji = "📚"
                    grade = "Average"
                    color = "#17a2b8"
                else:
                    emoji = "⚠️"
                    grade = "Needs Improvement"
                    color = "#dc3545"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                    <h1 style="font-size: 4rem;">{emoji}</h1>
                    <h2>GPA: {prediction:.2f} / 4.0</h2>
                    <h3>{grade}</h3>
                </div>
                """, unsafe_allow_html=True)
                
            with col_res2:
                st.markdown("#### 📊 GPA Scale:")
                st.progress(min(prediction/4.0, 1.0))
                
                st.markdown("""
                | GPA | Grade |
                |-----|--------|
                | 3.5 - 4.0 | Excellent |
                | 3.0 - 3.49 | Good |
                | 2.0 - 2.99 | Average |
                | < 2.0 | Needs Improvement |
                """)
            
            if prediction >= 3.5:
                st.balloons()
                
    except Exception as e:
        st.error(f"Error loading model: {e}")

elif page == "📊 Student Performance (std.csv)":
    st.markdown('<p class="main-header">📊 Student Performance Analysis</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 1. Load Dataset")
    
    try:
        import os
        st.markdown(f"Debug - Files in directory: {os.listdir('.')}")
        df = pd.read_csv('Datasets/std.csv')
        st.markdown("Dataset loaded successfully!")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.markdown(f"**Total Records:** {len(df)}")
        with col_info2:
            st.markdown(f"**Features:** {len(df.columns)}")
        
        st.markdown("### 2. Filter by Age")
        
        min_age = int(df['Student_Age'].min())
        max_age = int(df['Student_Age'].max())
        
        max_age_filter = st.slider("Select Maximum Age", min_age, max_age, max_age)
        
        df_filtered = df[df['Student_Age'] <= max_age_filter]
        
        st.markdown(f"**Filtered Records:** {len(df_filtered)} students (Age ≤ {max_age_filter})")
        
        st.markdown("---")
        st.markdown("### 3. Data Visualization")
        
        cols_to_encode = ['Sex', 'Additional_Work', 'Sports_activity', 'Transportation', 'Reading', 'Notes', 'Listening_in_Class', 'Project_work']
        
        df_viz = df_filtered.copy()
        for col in cols_to_encode:
            if col in df_viz.columns:
                df_viz[col] = df_viz[col].map({'Male': 0, 'Female': 1, 'Yes': 1, 'No': 0, 'Bus': 0, 'Private': 1})
        
        cols_to_show = ['Student_Age', 'Weekly_Study_Hours', 'Attendance Percentage', 'GPA']
        
        st.markdown("#### Average Values by Age")
        summary = df_viz.groupby('Student_Age')[cols_to_show].mean().reset_index()
        st.dataframe(summary)
        
        st.markdown("#### 📊 Study Hours by Age")
        st.bar_chart(summary.set_index('Student_Age')['Weekly_Study_Hours'])
        
        st.markdown("#### 📊 Attendance by Age")
        st.bar_chart(summary.set_index('Student_Age')['Attendance Percentage'])
        
        st.markdown("#### 📊 GPA by Age")
        st.bar_chart(summary.set_index('Student_Age')['GPA'])
        
        st.markdown("#### 📈 Statistics Summary")
        st.dataframe(df_viz[cols_to_show].describe())
        
        st.markdown("### 4. Prediction Models")
        
        model_type = st.selectbox("Select Model", ["Linear Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machines (SVM)"])
        
        if st.button("🔮 Train & Predict", type="primary"):
            with st.spinner("Training model..."):
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import LabelEncoder, StandardScaler
                from sklearn.linear_model import LinearRegression
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.svm import SVC
                from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
                
                df_model = df.copy()
                
                df_model['Sex'] = df_model['Sex'].map({'Male': 0, 'Female': 1})
                df_model['Additional_Work'] = df_model['Additional_Work'].map({'Yes': 1, 'No': 0})
                df_model['Sports_activity'] = df_model['Sports_activity'].map({'Yes': 1, 'No': 0})
                df_model['Transportation'] = df_model['Transportation'].map({'Bus': 0, 'Private': 1})
                df_model['Reading'] = df_model['Reading'].map({'Yes': 1, 'No': 0})
                df_model['Notes'] = df_model['Notes'].map({'Yes': 1, 'No': 0})
                df_model['Listening_in_Class'] = df_model['Listening_in_Class'].map({'Yes': 1, 'No': 0})
                df_model['Project_work'] = df_model['Project_work'].map({'Yes': 1, 'No': 0})
                
                df_model = df_model.dropna()
                
                X = df_model.drop(['GPA'], axis=1)
                y = df_model['GPA']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                if model_type == "Linear Regression":
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.markdown("#### Model Results: Linear Regression")
                    
                    import matplotlib.pyplot as plt
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.scatter(y_test, y_pred, alpha=0.6, c='blue', edgecolors='black')
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        ax.set_xlabel('Actual GPA')
                        ax.set_ylabel('Predicted GPA')
                        ax.set_title('Actual vs Predicted GPA')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col_res2:
                        st.markdown(f"**Mean Squared Error (MSE):** {mse:.4f}")
                        st.markdown(f"**R² Score:** {r2:.4f}")
                        st.markdown(f"**Root Mean Squared Error (RMSE):** {np.sqrt(mse):.4f}")
                    
                    results_df = pd.DataFrame({'Actual GPA': y_test.values, 'Predicted GPA': y_pred})
                    st.markdown("#### Prediction Results Table")
                    st.dataframe(results_df.head(10))
                    
                elif model_type == "K-Nearest Neighbors (KNN)":
                    y_class = (y * 2).astype(int)
                    y_train_class = (y_train * 2).astype(int)
                    y_test_class = (y_test * 2).astype(int)
                    
                    model = KNeighborsClassifier(n_neighbors=5)
                    model.fit(X_train_scaled, y_train_class)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test_class, y_pred)
                    
                    st.markdown("#### Model Results: K-Nearest Neighbors (KNN)")
                    
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    categories = ['Correct', 'Incorrect']
                    correct = int(acc * len(y_test_class))
                    incorrect = len(y_test_class) - correct
                    values = [correct, incorrect]
                    colors = ['#2ecc71', '#e74c3c']
                    ax.bar(categories, values, color=colors)
                    ax.set_ylabel('Count')
                    ax.set_title(f'KNN Classification Results (Accuracy: {acc:.2%})')
                    for i, v in enumerate(values):
                        ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)
                    st.pyplot(fig)
                    
                    st.markdown(f"**Accuracy:** {acc:.2%}")
                    
                elif model_type == "Support Vector Machines (SVM)":
                    y_class = (y * 2).astype(int)
                    y_train_class = (y_train * 2).astype(int)
                    y_test_class = (y_test * 2).astype(int)
                    
                    model = SVC(kernel='rbf', random_state=42)
                    model.fit(X_train_scaled, y_train_class)
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test_class, y_pred)
                    
                    st.markdown("#### Model Results: Support Vector Machines (SVM)")
                    
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    categories = ['Correct', 'Incorrect']
                    correct = int(acc * len(y_test_class))
                    incorrect = len(y_test_class) - correct
                    values = [correct, incorrect]
                    colors = ['#3498db', '#e74c3c']
                    ax.bar(categories, values, color=colors)
                    ax.set_ylabel('Count')
                    ax.set_title(f'SVM Classification Results (Accuracy: {acc:.2%})')
                    for i, v in enumerate(values):
                        ax.text(i, v + 0.5, str(v), ha='center', fontsize=12)
                    st.pyplot(fig)
                    
                    st.markdown(f"**Accuracy:** {acc:.2%}")
                
                st.success("Model trained successfully!")
        
    except Exception as e:
        st.error(f"Error: {e}")

elif page == "🔥 Train Iris Neural Network":
    st.markdown('<p class="main-header">🔥 Train Iris Neural Network</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 1. Hyperparameters Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hidden_layer_size = st.number_input("Hidden Layer Size", min_value=16, max_value=128, value=64, step=16)
    with col2:
        num_epochs = st.number_input("Number of Epochs", min_value=50, max_value=200, value=100, step=10)
    with col3:
        learning_rate = st.number_input("Learning Rate", min_value=0.00, max_value=0.01, value=0.001, step=0.001, format="%.3f")
    
    st.markdown("---")
    st.markdown("### 2. Load & Prepare Data")
    
    try:
        df_iris = pd.read_csv('Datasets/iris_dirty.csv')
        
        st.markdown("#### Raw Data (with issues):")
        st.dataframe(df_iris.head(10))
        
        st.markdown("#### Data Cleaning:")
        
        df_clean = df_iris.copy()
        df_clean = df_clean.drop('Unnamed: 0', axis=1, errors='ignore')
        
        df_clean['Sepal.Length'] = pd.to_numeric(df_clean['Sepal.Length'], errors='coerce')
        df_clean['Sepal.Width'] = pd.to_numeric(df_clean['Sepal.Width'], errors='coerce')
        df_clean['Petal.Length'] = pd.to_numeric(df_clean['Petal.Length'], errors='coerce')
        df_clean['Petal.Width'] = pd.to_numeric(df_clean['Petal.Width'], errors='coerce')
        
        df_clean['Sepal.Length'].fillna(df_clean['Sepal.Length'].median(), inplace=True)
        df_clean['Sepal.Width'].fillna(df_clean['Sepal.Width'].median(), inplace=True)
        df_clean['Petal.Length'].fillna(df_clean['Petal.Length'].median(), inplace=True)
        df_clean['Petal.Width'].fillna(df_clean['Petal.Width'].median(), inplace=True)
        
        df_clean['Species'] = df_clean['Species'].str.lower()
        
        le = LabelEncoder()
        df_clean['Species_encoded'] = le.fit_transform(df_clean['Species'])
        
        st.success(f"Cleaned {df_iris.isnull().sum().sum()} missing values")
        st.success("Normalized species labels to lowercase")
        
        X = df_clean.drop(['Species', 'Species_encoded'], axis=1)
        y = df_clean['Species_encoded']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        st.markdown(f"**Training samples:** {len(X_train)}")
        st.markdown(f"**Testing samples:** {len(X_test)}")
        
        st.markdown("---")
        st.markdown("### 3. Train Model")
        
        if st.button("🚀 Start Training", type="primary"):
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.utils import to_categorical
            
            model = Sequential([
                Dense(hidden_layer_size, activation='relu', input_shape=(4,)),
                Dense(hidden_layer_size // 2, activation='relu'),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            y_train_cat = to_categorical(y_train, 3)
            y_test_cat = to_categorical(y_test, 3)
            
            st.markdown("---")
            st.markdown("### 4. Training Logs")
            
            logs_container = st.empty()
            logs_text = ""
            
            epochs_list = []
            train_loss_list = []
            val_loss_list = []
            train_acc_list = []
            val_acc_list = []
            
            progress_bar = st.progress(0)
            
            for epoch in range(1, num_epochs + 1):
                history = model.fit(
                    X_train, y_train_cat,
                    epochs=1,
                    batch_size=16,
                    validation_data=(X_test, y_test_cat),
                    verbose=0
                )
                
                train_loss = history.history['loss'][0]
                val_loss = history.history['val_loss'][0]
                train_acc = history.history['accuracy'][0]
                val_acc = history.history['val_accuracy'][0]
                
                epochs_list.append(epoch)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                train_acc_list.append(train_acc * 100)
                val_acc_list.append(val_acc * 100)
                
                if epoch % 10 == 0 or epoch == num_epochs:
                    log_msg = f"Epoch [{epoch}/{num_epochs}] - Train Accuracy: {train_acc*100:.2f}% | Validation Accuracy: {val_acc*100:.2f}%"
                    logs_text += log_msg + "\n"
                    logs_container.text_area("Training Logs", logs_text, height=200)
                
                progress_bar.progress(epoch / num_epochs)
            
            st.markdown("---")
            st.markdown("### 5. Final Accuracy")
            
            final_train_acc = train_acc_list[-1]
            final_val_acc = val_acc_list[-1]
            
            st.markdown(f"**Training Accuracy:** {final_train_acc:.2f}%")
            st.markdown(f"**Validation Accuracy:** {final_val_acc:.2f}%")
            
            st.markdown("---")
            st.markdown("### 6. Training & Validation Loss and Accuracy")
            
            chart_data = pd.DataFrame({
                'Epoch': epochs_list,
                'Training Loss': train_loss_list,
                'Validation Loss': val_loss_list,
                'Training Accuracy': train_acc_list,
                'Validation Accuracy': val_acc_list
            })
            
            st.line_chart(chart_data.set_index('Epoch'))
            
            st.markdown("""
            **Legend:**
            - 🔵 Training Loss (Dark Blue)
            - 🔷 Validation Loss (Light Blue)
            - 🔴 Training Accuracy (Red)
            - 🔺 Validation Accuracy (Pink)
            """)
            
            st.success("Training completed successfully!")
            
    except Exception as e:
        st.error(f"Error: {e}")
