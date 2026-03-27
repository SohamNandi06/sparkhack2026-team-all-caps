import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from model.net import Net


def load_model():
    model = Net()

    try:
        model.load_state_dict(torch.load("saved_model/global_model.pth"))
        print("✅ Loaded trained global model")
    except Exception as e:
        print("❌ Error loading model:", e)
        print("⚠️ Make sure you ran the server training first")

    model.eval()
    return model



def preprocess(sample):
    
    df = pd.read_csv("data/heart_disease.csv", header=None)

  
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)

    X = df.iloc[:, 0:13].values

    scaler = StandardScaler()
    scaler.fit(X)

    return scaler.transform(sample.reshape(1, -1))


def predict(model, sample):
    x = torch.tensor(sample, dtype=torch.float32)

    with torch.no_grad():
        output = model(x)
        prob = output.item()
        pred = 1 if prob > 0.5 else 0

    return prob, pred


def get_user_input():
    print("\nEnter 13 features (comma-separated):")
    print("Example:")
    print("52,1,2,130,250,0,1,140,0,1.2,2,0,2\n")

    user_input = input("Input: ")

    try:
        values = [float(x.strip()) for x in user_input.split(",")]

        if len(values) != 13:
            raise ValueError("You must enter exactly 13 values.")

        return np.array(values)

    except Exception as e:
        print("❌ Invalid input:", e)
        return None


if __name__ == "__main__":
    print("\n🔍 Federated Model — Live Prediction Demo\n")

    model = load_model()

    sample = get_user_input()

    if sample is not None:
        processed = preprocess(sample)

        prob, pred = predict(model, processed)

        print("\n📊 Prediction Result:")
        print(f"Probability of Heart Disease: {prob:.4f}")

        if pred == 1:
            print("⚠️ HIGH RISK: Heart Disease Detected")
        else:
            print("✅ LOW RISK: No Heart Disease")

    else:
        print("❌ Prediction aborted due to invalid input")