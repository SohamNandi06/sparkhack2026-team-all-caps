import flwr as fl
import json
import os
import torch

from flwr.common import parameters_to_ndarrays
from model.net import Net
from clients.data_utils import load_full_data


class CustomStrategy(fl.server.strategy.FedAvg):

    def __init__(self):
        super().__init__()
        self.latest_parameters = None 

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        self.latest_parameters = aggregated_parameters

        print(f"Round {server_round} training aggregated")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}

        losses = []
        individual_accs = []

        for _, res in results:
            losses.append(res.loss)

            if "accuracy" in res.metrics:
                individual_accs.append(res.metrics["accuracy"])

        avg_loss = sum(losses) / len(losses)

        X_test, y_test = load_full_data()

        model = Net()

        if self.latest_parameters is not None:
            ndarrays = parameters_to_ndarrays(self.latest_parameters)

            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict} 
            model.load_state_dict(state_dict, strict=True)

        model.eval()

        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        with torch.no_grad():
            outputs = model(X_tensor)
            preds = (outputs > 0.5).float()
            global_acc = (preds == y_tensor).float().mean().item()

        log_data = {
            "round": server_round,
            "accuracy": global_acc,
            "loss": avg_loss,
            "hospitals": len(results),
            "individual_accuracies": individual_accs
        }

        os.makedirs("logs", exist_ok=True)
        log_file = "logs/rounds.json"

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(log_data)

        with open(log_file, "w") as f:
            json.dump(data, f, indent=4)

        print(
            f"Round {server_round} → Global Acc: {global_acc:.4f}, Loss: {avg_loss:.4f}"
        )
        print(f"Individual Accuracies: {individual_accs}")

        return avg_loss, {"accuracy": global_acc}


if __name__ == "__main__":
    strategy = CustomStrategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )