import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler

def preprocess():
    input_file = "cpu_usage.csv"
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "processed_data.csv")
    scaler_file = os.path.join(output_dir, "scaler.pkl")
    
    df = pd.read_csv(input_file)
    
    features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
    target = 'cpu_usage'
    
    df = df[features + [target]].copy()
    
    df = df.dropna()
    
    if 'controller_kind' in df.columns:
        df = pd.get_dummies(df, columns=['controller_kind'], drop_first=False)
    
    numeric_features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    
    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessing done. Saved to {output_file}")
    print(f"✅ Scaler saved to {scaler_file}")

if __name__ == "__main__":
    preprocess()