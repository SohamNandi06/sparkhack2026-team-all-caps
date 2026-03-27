import flwr as fl
import numpy as np
import torch
from torch import nn, optim
import random

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

from model.net import Net
from clients.data_utils import load_partition
from privacy.dp_utils import add_dp_noise
from privacy.encrypt import encrypt


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        print(f"\n🏥 Initializing Hospital {hospital_id}")

        self.X_train, self.X_test, self.y_train, self.y_test = load_partition(hospital_id)

        print(f"[Hospital {hospital_id}] Local dataset loaded")
        print(f"[Hospital {hospital_id}] Train shape: {self.X_train.shape}")
        print(f"[Hospital {hospital_id}] Test shape: {self.X_test.shape}")

        self.model = Net()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"\n[Hospital {self.hospital_id}] Training locally...")

        self.set_parameters(parameters)

        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1)

        self.model.train()

        for epoch in range(1):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

        print(f"[Hospital {self.hospital_id}] Training complete")

        weights = self.get_parameters(config)

        weights = add_dp_noise(weights)

        encrypted_weights = encrypt(weights)

        print(f"[Hospital {self.hospital_id}] Sending encrypted model updates (NOT raw data)")

        return weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        print(f"[Hospital {self.hospital_id}] Evaluating locally...")

        self.set_parameters(parameters)

        X_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_test, dtype=torch.float32).view(-1, 1)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            preds = (outputs > 0.5).float()
            acc = (preds == y_tensor).float().mean().item()

        print(f"[Hospital {self.hospital_id}] Local Accuracy: {acc:.4f}")

        return float(loss), len(self.X_test), {"accuracy": acc}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital-id", type=int, required=True)
    args = parser.parse_args()

    client = FlowerClient(args.hospital_id)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
    )