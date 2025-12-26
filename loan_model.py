import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, auc)

def load_and_clean_data(path="loan_data.csv"):
    df = pd.read_csv(path)

    # Cleaning
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_exp'] <= 50]
    df['person_income'] = df['person_income'].clip(upper=1000000)
    df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes':1, 'No':0})

    return df

def encode_features(df):
    le = LabelEncoder()
    df['person_education'] = le.fit_transform(df['person_education'])

    one_hot_cols = ['person_gender', 'loan_intent', 'person_home_ownership']
    df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

    # Convert boolean to int
    bool_cols = df.select_dtypes('bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df

def split_data(df, test_size=0.3):
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_numeric(X_train, X_test):
    num_cols = ['person_age','person_income','person_emp_exp','loan_amnt',
                'loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score']

    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, scaler

def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    neg = y_train.value_counts()[0]
    pos = y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.5,
        reg_lambda=1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(f"\n--- {model_name} Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print(f"{model_name} ROC-AUC:", roc_auc)
    print(f"{model_name} PR-AUC:", pr_auc)
    return y_pred, y_prob

def save_model(model, scaler, model_path="xgb_loan_model.pkl", scaler_path="scaler.pkl"):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("Model and scaler saved successfully!")

def main():
    df = load_and_clean_data()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = scale_numeric(X_train, X_test)

    # Logistic Regression
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Save XGBoost model and scaler
    save_model(xgb_model, scaler)


if __name__ == "__main__":
    main()
