import argparse
from functools import partial
from typing import Any

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from fl.aml_model_fedper import AMLNet
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config, make_dict_with_epochs_or_steps
from fl4health.utils.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn

import torch
import numpy as np
from flwr.common.parameter import ndarrays_to_parameters

from fl4health.checkpointing.checkpointer import BestLossTorchModuleCheckpointer, LatestTorchModuleCheckpointer
from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule

def get_base_module_parameters(model: torch.nn.Module) -> fl.common.Parameters:
    """
    Extract parameters from the base_module part of the model as NumPy arrays.
    
    Args:
        model: The model containing a base_module attribute
        
    Returns:
        Parameters object with base_module parameters
    """
    # Get state dict of base module
    state_dict = model.base_module.state_dict()
    
    # Convert parameters to numpy arrays in the same order as state_dict
    params = [param.detach().cpu().numpy() for param in state_dict.values()]
    
    # Convert to Parameters
    return ndarrays_to_parameters(params)


def fit_config(
    batch_size: int,
    n_server_rounds: int,
    downsampling_ratio: float,
    current_round: int,
    local_epochs: int | None = None,
    local_steps: int | None = None,
) -> Config:
    return {
        **make_dict_with_epochs_or_steps(local_epochs, local_steps),
        "batch_size": batch_size,
        "n_server_rounds": n_server_rounds,
        "downsampling_ratio": downsampling_ratio,
        "current_server_round": current_round,
    }


def main(config: dict[str, Any]) -> None:
    # Function to produce config for each client
    fit_config_fn = partial(
        fit_config,
        config["batch_size"],
        config["n_server_rounds"],
        config["downsampling_ratio"],
        local_epochs=config.get("local_epochs"),
        local_steps=config.get("local_steps"),
    )

    # Initialize your model
    full_model = AMLNet()

    # Wrap it for FedPer
    model = SequentiallySplitExchangeBaseModel(
        base_module=full_model.base_module,  # Base layers for global aggregation
        head_module=full_model.head_module   # Head layers personalized per client
    )

    # 3. Create parameter exchanger (IMPORTANT: only base_module parameters)
    from fl4health.parameter_exchange.partial_parameter_exchanger import PartialParameterExchanger
    from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames
    from fl.utils.FedPerExchanger import FedPerExchanger

    # parameter_exchanger = PartialParameterExchanger(parameter_packer=ParameterPackerWithLayerNames())
    parameter_exchanger = FedPerExchanger()

    checkpointers = [
        BestLossTorchModuleCheckpointer(config["checkpoint_path"], "best_model_fedper.pkl"),
        LatestTorchModuleCheckpointer(config["checkpoint_path"], "latest_model_fedper.pkl"),
    ]

    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model,
        parameter_exchanger=parameter_exchanger,
        model_checkpointers=checkpointers,
    )    

    checkpoint_and_state_module = BaseServerCheckpointAndStateModule(
        model=model,
        parameter_exchanger=parameter_exchanger,
        model_checkpointers=checkpointers
    )    


    # Server strategy
    strategy = FedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters = parameter_exchanger.push_parameters(model, config={}),
        # initial_parameters=get_base_module_parameters(model)  # Only send base layers
    )

    client_manager = SimpleClientManager()
    server = FlServer(
        client_manager=client_manager,
        fl_config=config,
        strategy=strategy,
        checkpoint_and_state_module=checkpoint_and_state_module,
        accept_failures=False,
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FedPer FL Server")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="fl/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)