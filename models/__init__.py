"""models — neural network architectures for fl-privacy-project."""

from models.base_model import BaseModel
from models.mlp import MLP
from models.cnn import CNN1D


def get_model(name: str, input_dim: int, num_classes: int, **kwargs) -> BaseModel:
    """
    Model factory.

    Parameters
    ----------
    name        : ``'mlp'`` or ``'cnn'``
    input_dim   : number of input features
    num_classes : number of output classes
    **kwargs    : forwarded to the model constructor (hidden_dims, dropout, …)

    Returns
    -------
    BaseModel instance (nn.Module subclass)
    """
    name = name.lower()
    if name == "mlp":
        hidden_dims = kwargs.get("hidden_dims", [256, 128])
        dropout = kwargs.get("dropout", 0.3)
        return MLP(input_dim=input_dim, hidden_dims=hidden_dims,
                   num_classes=num_classes, dropout=dropout)
    if name == "cnn":
        in_channels = kwargs.get("in_channels", 1)
        return CNN1D(in_channels=in_channels, seq_len=input_dim,
                     num_classes=num_classes)
    raise ValueError(f"Unknown model: '{name}'.  Choose 'mlp' or 'cnn'.")


__all__ = ["BaseModel", "MLP", "CNN1D", "get_model"]
