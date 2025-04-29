# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. The goal is to build an effective binary classification model to distinguish between legitimate and fraudulent activities.

## Project Overview

Credit card fraud has become a significant issue in the financial industry. Detecting fraudulent transactions as early as possible is critical to reducing financial losses. In this project, I have used a public dataset and applied data preprocessing, logistic regression modeling, and evaluation techniques to build a fraud detection system.

## Dataset

- The dataset consists of real-world credit card transactions.
- It contains 31 features, including anonymized PCA components (V1 to V28), transaction amount, and the class label.
- The dataset is highly imbalanced: approximately 0.2% of the transactions are fraudulent.

## Approach

- **Data Exploration:** Visualized class distribution and transaction amounts.
- **Data Balancing:** Applied undersampling to create a balanced dataset for training.
- **Model Building:** Trained a Logistic Regression model using scikit-learn.
- **Evaluation:** Evaluated the model using confusion matrix, accuracy, precision, and recall metrics.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib (for model saving)

## How to Run

1. Clone this repository.
2. Install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
3. Run the Jupyter Notebook or Python script provided to train the model and evaluate results.
4. Optionally, run the Streamlit app to interactively predict fraud based on new data:
    ```bash
    streamlit run app.py
    ```

## Project Highlights

- Balanced dataset before training to avoid biased results.
- Achieved strong classification performance even on a difficult, real-world imbalanced dataset.
- Saved the trained model for future predictions.

## Results

- High precision and recall scores achieved for fraud detection.
- Confusion matrix visualization shows effective fraud classification with minimal false negatives.

## Future Work

- Explore more advanced algorithms like Random Forest, XGBoost, or ensemble methods.
- Implement real-time fraud detection system deployment.
- Use SMOTE (Synthetic Minority Oversampling) instead of undersampling for better generalization.

## Acknowledgements

- Thanks to the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) for providing the data used in this project.
