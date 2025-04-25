import argparse
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config, NDArrays
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl4health.clients.fedper_client import FedPerClient
from fl.utils.FedPerExchanger import FedPerExchanger

from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.utils.config import load_config, narrow_dict_type
from fl4health.utils.metrics import BalancedAccuracy

from fl.aml_model_fedper import AMLNet
from fl.load_aml_data_V01 import load_aml_data


class AMLFedPerClient(FedPerClient):
    def __init__(self, data_path: Path, metrics, device: torch.device) -> None:
        super().__init__(data_path=data_path, metrics=metrics, device=device)
        self.parameter_exchanger = None

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        bank_filter = config.get("bank_filter", None)
        train_loader, val_loader, _, pos_weight = load_aml_data(self.data_path, batch_size, bank_filter)
        self.positive_weight = pos_weight
        return train_loader, val_loader

    def get_test_data_loader(self, config: Config) -> DataLoader | None:
        batch_size = narrow_dict_type(config, "batch_size", int)
        bank_filter = config.get("bank_filter", None)
        _, _, test_loader, _ = load_aml_data(self.data_path, batch_size, bank_filter)
        return test_loader

    def get_model(self, config: Config) -> nn.Module:
        # Create the AMLNet model
        aml_model = AMLNet()
        # Wrap inside SequentiallySplitExchangeBaseModel for FedPer
        return SequentiallySplitExchangeBaseModel(
            base_module=aml_model.base_module,
            head_module=aml_model.head_module
        ).to(self.device)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_criterion(self, config: Config) -> _Loss:
        positive_weight_value = getattr(self, 'positive_weight', 1.0)
        pos_weight_tensor = torch.tensor([positive_weight_value], dtype=torch.float32).to(self.device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    def get_parameter_exchanger(self, config: Config) -> FedPerExchanger:
        if self.parameter_exchanger is None:
            self.parameter_exchanger = FedPerExchanger()
        return self.parameter_exchanger
    
    # Override the set_parameters method to use our custom parameter exchanger
    def set_parameters(self, parameters, config, fitting_round=False):
        if fitting_round and not hasattr(self, "model"):
            self.initialize_model_and_data(config)
            
        if self.parameter_exchanger is None:
            self.parameter_exchanger = self.get_parameter_exchanger(config)
            
        # Use our custom parameter exchanger
        self.parameter_exchanger.pull_parameters(parameters, self.model, config)
    
    # Override fit to ensure it returns the correct format
    def fit(self, parameters, config):
        # Call the parent's training logic with our parameters
        super().fit(parameters, config)
        
        # Now get the updated parameters, sample count, and metrics
        if self.parameter_exchanger is None:
            self.parameter_exchanger = self.get_parameter_exchanger(config)
            
        # Get updated base module parameters
        updated_parameters = self.parameter_exchanger.push_parameters(self.model, None, config)
        
        # Convert to NDArrays if needed
        if hasattr(updated_parameters, 'tensors'):
            from flwr.common.parameter import parameters_to_ndarrays
            updated_parameters = parameters_to_ndarrays(updated_parameters)
            
        # Sample count (number of examples used for training)
        sample_count = len(self.train_loader.dataset) if hasattr(self, 'train_loader') else 0
        
        # Metrics dictionary - use the metrics from the last training round
        metrics = {}
        if hasattr(self, 'metrics_cache') and 'train' in self.metrics_cache:
            metrics = self.metrics_cache['train']
        
        # Return the tuple with 3 elements that Flower expects
        return updated_parameters, sample_count, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedPer AML Client")
    parser.add_argument("--config_path", type=str, default="fl/config.yaml", help="Path to config file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the AML dataset")
    parser.add_argument("--bank_filter", type=int, default=None, help="Restrict to data from a single bank")
    parser.add_argument("--positive_weight", type=float, default=1.0, help="Positive class weight")
    args = parser.parse_args()

    config = load_config(args.config_path)
    config["bank_filter"] = args.bank_filter
    config["positive_weight"] = args.positive_weight

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)

    client = AMLFedPerClient(data_path, [BalancedAccuracy("balanced_accuracy")], device)
    fl.client.start_client(server_address="localhost:8080", client=client.to_client())
    client.shutdown()