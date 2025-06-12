import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            color: #31333f;
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #0056b3;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Credit Card Fraud Detection App")

st.markdown("This application helps in detecting fraudulent transactions using a **Logistic Regression** model.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["📊 Data Overview", "📈 Visualizations", "🧠 Predict Fraud"])

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if option == "📊 Data Overview":
        st.header("📊 Data Overview")
        st.subheader("Top 5 Records")
        st.dataframe(df.head())

        st.subheader("Dataset Information")
        st.write(df.info())

        st.subheader("Null Values")
        st.write(df.isnull().sum())

        st.subheader("Class Distribution")
        st.write(df['Class'].value_counts())
        st.bar_chart(df['Class'].value_counts())

    elif option == "📈 Visualizations":
        st.header("📈 Fraud Detection Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Class Distribution (Pie Chart)")
            fig1, ax1 = plt.subplots()
            df['Class'].value_counts().plot.pie(autopct='%1.1f%%', colors=["#36a2eb", "#ff6384"], ax=ax1)
            ax1.set_ylabel('')
            st.pyplot(fig1)

        with col2:
            st.subheader("Amount Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df['Amount'], bins=30, kde=True, color='#4bc0c0', ax=ax2)
            st.pyplot(fig2)

        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", linewidths=0.5, ax=ax3)
        st.pyplot(fig3)

        st.subheader("V1 vs Amount (Scatterplot)")
        fig4, ax4 = plt.subplots()
        sns.scatterplot(data=df, x='V1', y='Amount', hue='Class', palette='viridis', alpha=0.6, ax=ax4)
        st.pyplot(fig4)

    elif option == "🧠 Predict Fraud":
        st.header("🧠 Predict on Uploaded Dataset")
        if 'Class' in df.columns:
            X = df.drop(columns=['Class'])
            y_true = df['Class']
        else:
            X = df
            y_true = None

        y_pred = model.predict(X)

        st.subheader("Prediction Results")
        result_df = df.copy()
        result_df['Prediction'] = y_pred
        st.dataframe(result_df.head(10))

        if y_true is not None:
            st.subheader("Evaluation Metrics")
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            st.write(cm)

            st.write("Classification Report:")
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

        st.success("✅ Prediction Completed!")

else:
    st.info("Please upload a CSV file from the sidebar to get started.")
