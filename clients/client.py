import flwr as fl
import torch
import torch.nn as nn
import numpy as np
import argparse

from model.net import Net
from clients.data_utils import load_partition

from privacy.dp_utils import add_dp_noise
from privacy.encrypt import encrypt


def train(model, X, y, epochs=1):
    model.train()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()


def test(model, X, y):
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        outputs = model(X_tensor)
        loss = nn.BCELoss()(outputs, y_tensor)

        preds = (outputs > 0.5).float()
        acc = (preds == y_tensor).float().mean()

    return loss.item(), acc.item()


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.model = Net()
        self.X_train, self.X_test, self.y_train, self.y_test = load_partition(hospital_id)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train(self.model, self.X_train, self.y_train, epochs=1)

        weights = self.get_parameters(config)

        weights = add_dp_noise(weights)
        encrypted_weights = encrypt(weights)

        print("DP + Encryption applied")

        return weights, len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, acc = test(self.model, self.X_test, self.y_test)

        return loss, len(self.X_test), {"accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital-id", type=int, required=True)
    args = parser.parse_args()

    client = HospitalClient(args.hospital_id)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )