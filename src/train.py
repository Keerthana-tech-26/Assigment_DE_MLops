import sys
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train(input_file, model_file):
    df = pd.read_csv(input_file)
    X = df.drop('cpu_usage', axis=1)
    y = df['cpu_usage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
