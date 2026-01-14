import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_random_forest_model(n_estimators=100, min_samples_split=50, random_state=1):
    """
    Create a Random Forest Classifier with specified parameters.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, random_state=random_state)
    return model

def train_model(model, X_train, y_train):
    """
    Train a machine learning model.
    """
    model.fit(X_train, y_train)
    return model

def predict(train, test, predictors, model):
    """
    Make predictions on test data.
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def predict_with_threshold(train, test, predictors, model, threshold=0.6):
    """
    Make predictions with probability threshold.
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= threshold] = 1
    preds[preds < threshold] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    """
    Perform walk-forward backtesting.
    """
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def backtest_with_threshold(data, model, predictors, start=2500, step=250, threshold=0.6):
    """
    Perform walk-forward backtesting with probability threshold.
    """
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict_with_threshold(train, test, predictors, model, threshold)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def evaluate_model(predictions):
    """
    Evaluate model performance.
    """
    accuracy = accuracy_score(predictions["Target"], predictions["Predictions"])
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'prediction_counts': predictions["Predictions"].value_counts().to_dict(),
        'target_distribution': predictions["Target"].value_counts(normalize=True).to_dict()}
    return results

def print_model_results(predictions, model_name="Model"):
    """
    Print model evaluation results in a simplified format.
    """
    results = evaluate_model(predictions)
    print(f"Accuracy of {model_name} on training set: {results['accuracy']:.2f}")

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance from a trained model.
    """
    #Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    #Plot top N features
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'].values)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def compare_models(predictions_dict):
    """
    Compare multiple models' performance.
    """
    results = []
    for model_name, predictions in predictions_dict.items():
        metrics = evaluate_model(predictions)
        results.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}"})
    comparison_df = pd.DataFrame(results)
    return comparison_df