import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.metrics import Accuracy
from fl4health.utils.config import load_config

from fl.aml_model import AMLNet  # Your model
from fl.load_aml_data import load_aml_data

class AMLClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        bank_filter = config.get("bank_filter", None)
        train_loader, val_loader, _ = load_aml_data(self.data_path, batch_size, bank_filter)
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        bank_filter = config.get("bank_filter", None)
        _, _, test_loader = load_aml_data(self.data_path, batch_size, bank_filter)
        return test_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.BCEWithLogitsLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_model(self, config: Config) -> nn.Module:
        input_dim = narrow_dict_type(config, "input_dim", int)
        return AMLNet(input_dim).to(self.device)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="FL AML Client")
    # parser.add_argument("--dataset_path", action="store", type=str, help="Path to the AML dataset")
    # parser.add_argument("--input_dim", type=int, default=12, help="Input feature count")
    # args = parser.parse_args()
    parser = argparse.ArgumentParser(description="FL AML Client")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the AML dataset")
    parser.add_argument("--bank_filter", type=int, default=None, help="Restrict to data from a single bank")

    args = parser.parse_args()    

    config = load_config(args.config_path)
    config["bank_filter"] = args.bank_filter 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    # config = {"input_dim": args.input_dim, "batch_size": 1024}


    client = AMLClient(data_path, [Accuracy("accuracy")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
