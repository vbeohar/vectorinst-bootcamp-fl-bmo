from fl4health.parameter_exchange.partial_parameter_exchanger import PartialParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithLayerNames
from flwr.common.typing import Config, NDArrays
import torch.nn as nn
import torch
import numpy as np
from flwr.common.typing import Parameters
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays


class FedPerExchanger(PartialParameterExchanger[list[str]]):
    def __init__(self) -> None:
        super().__init__(parameter_packer=ParameterPackerWithLayerNames())

    def push_parameters(self, model: nn.Module, initial_model: nn.Module | None = None, config: Config | None = None) -> Parameters:
        """Push (send) parameters to the server (only base_module)."""
        base_state_dict = model.base_module.state_dict()
        model_weights = [param.detach().cpu().numpy() for param in base_state_dict.values()]
        return ndarrays_to_parameters(model_weights)     

    def pull_parameters(self, parameters: Parameters, model: nn.Module, config: Config | None = None) -> None:
        """Pull (receive) parameters from the server and update base_module."""
        # Check if parameters is already a list of NumPy arrays or a Parameters object
        if isinstance(parameters, list):
            numpy_params = parameters  # Already a list of NumPy arrays
        else:
            # It's a Parameters object, convert to NumPy arrays
            numpy_params = parameters_to_ndarrays(parameters)
        
        # Get current state dict
        base_state_dict = model.base_module.state_dict()
        
        # Create new state dict with received parameters
        if len(numpy_params) == len(base_state_dict):
            new_state_dict = {}
            for i, (key, _) in enumerate(base_state_dict.items()):
                new_state_dict[key] = torch.tensor(numpy_params[i])
                
            # Load state dict
            model.base_module.load_state_dict(new_state_dict, strict=True)
        else:
            raise ValueError(f"Parameter count mismatch: {len(numpy_params)} vs {len(base_state_dict)}")

    def select_parameters(self, model: nn.Module, initial_model: nn.Module | None = None) -> tuple[NDArrays, None]:
        # Extract parameters as numpy arrays in the same order as state_dict
        state_dict = model.base_module.state_dict()
        weights = [param.detach().cpu().numpy() for param in state_dict.values()]
        
        # Don't return additional info - just the parameters
        return weights, None

    def unpack_parameters(self, parameters: NDArrays) -> tuple[NDArrays, None]:
        # Simply return the parameters without unpacking
        return parameters, None