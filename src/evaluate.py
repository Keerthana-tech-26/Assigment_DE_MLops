import sys
import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

def evaluate(model_file="data/model.pkl", input_file="data/processed_data.csv", metrics_file="metrics.json"):
    df = pd.read_csv(input_file)
    X = df.drop('cpu_usage', axis=1)
    y = df['cpu_usage']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "train_mse": float(mean_squared_error(y_train, y_pred_train)),
        "test_mse": float(mean_squared_error(y_test, y_pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "train_accuracy_percent": float(r2_score(y_train, y_pred_train) * 100),
        "test_accuracy_percent": float(r2_score(y_test, y_pred_test) * 100)
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Evaluation Complete!")
    print(f"=" * 50)
    print(f"Test Metrics:")
    print(f"  - R² Score (Accuracy): {metrics['test_r2']:.4f} ({metrics['test_accuracy_percent']:.2f}%)")
    print(f"  - RMSE: {metrics['test_rmse']:.4f}")
    print(f"  - MAE: {metrics['test_mae']:.4f}")
    print(f"  - MSE: {metrics['test_mse']:.4f}")
    print(f"\nTraining Metrics:")
    print(f"  - R² Score: {metrics['train_r2']:.4f} ({metrics['train_accuracy_percent']:.2f}%)")
    print(f"  - RMSE: {metrics['train_rmse']:.4f}")
    print(f"=" * 50)

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])