import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
import sys

def preprocess():
    try:
        input_file = "cpu_usage.csv"
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "processed_data.csv")
        scaler_file = os.path.join(output_dir, "scaler.pkl")
        
        print(f"Reading data from {input_file}...")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{input_file} not found!")
        
        df = pd.read_csv(input_file)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes', 'controller_kind']
        target = 'cpu_usage'
        
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        df = df[features + [target]].copy()
        
        print(f"Dropping NaN values...")
        initial_rows = len(df)
        df = df.dropna()
        print(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        if 'controller_kind' in df.columns:
            print("Encoding controller_kind...")
            df = pd.get_dummies(df, columns=['controller_kind'], drop_first=False)
        
        numeric_features = ['cpu_request', 'mem_request', 'cpu_limit', 'mem_limit', 'runtime_minutes']
        
        print("Scaling numeric features...")
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        print(f"Saving scaler to {scaler_file}...")
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"Saving processed data to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print(f"Preprocessing done. Saved to {output_file}")
        print(f"Scaler saved to {scaler_file}")
        print(f"Final data shape: {df.shape}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    preprocess()