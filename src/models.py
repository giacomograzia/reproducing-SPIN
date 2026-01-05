"""
Model architectures and related utilities.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class BaseModel(nn.Module):
    """Base neural network model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.2,
        activation: str = "relu"
    ):
        """
        Initialize base model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function name
        """
        super(BaseModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Select activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)


class FairnessAwareModel(BaseModel):
    """Model with fairness constraints."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        sensitive_dim: int,
        dropout: float = 0.2,
        activation: str = "relu",
        fairness_weight: float = 0.1
    ):
        """
        Initialize fairness-aware model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            sensitive_dim: Dimension of sensitive attributes
            dropout: Dropout probability
            activation: Activation function name
            fairness_weight: Weight for fairness loss
        """
        super(FairnessAwareModel, self).__init__(
            input_dim, hidden_dims, output_dim, dropout, activation
        )
        
        self.sensitive_dim = sensitive_dim
        self.fairness_weight = fairness_weight
        
        # Adversarial network for fairness
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, sensitive_dim)
        )
    
    def forward(
        self, x: torch.Tensor, return_adversary: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with optional adversary output.
        
        Args:
            x: Input tensor
            return_adversary: Whether to return adversary prediction
            
        Returns:
            Model output (and adversary output if requested)
        """
        # Forward through all but last layer to get intermediate representation
        intermediate = x
        for layer in self.model[:-1]:
            intermediate = layer(intermediate)
        
        # Final output layer
        output = self.model[-1](intermediate)
        
        if return_adversary:
            adversary_output = self.adversary(intermediate)
            return output, adversary_output
        
        return output


def create_model(config: dict, model_type: str = "base") -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config: Configuration dictionary
        model_type: Type of model to create
        
    Returns:
        Model instance
        
    Raises:
        KeyError: If required config keys are missing
        ValueError: If model_type is unknown
    """
    if "model" not in config:
        raise KeyError("'model' key not found in config")
    
    model_config = config["model"]
    
    if model_type == "base":
        model = BaseModel(
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            output_dim=model_config["output_dim"],
            dropout=model_config["dropout"],
            activation=model_config["activation"]
        )
    elif model_type == "fairness":
        if "fairness" not in config:
            raise KeyError("'fairness' key not found in config for fairness model")
        if "sensitive_attributes" not in config["fairness"]:
            raise KeyError("'sensitive_attributes' not found in config['fairness']")
        
        model = FairnessAwareModel(
            input_dim=model_config["input_dim"],
            hidden_dims=model_config["hidden_dims"],
            output_dim=model_config["output_dim"],
            sensitive_dim=len(config["fairness"]["sensitive_attributes"]),
            dropout=model_config["dropout"],
            activation=model_config["activation"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model
