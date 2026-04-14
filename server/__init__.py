"""server — federated aggregation and orchestration."""

from server.server import FLServer, get_server
from server.aggregation import fedavg, fedmedian, weighted_fedavg
from server.ensemble import EnsembleServer

__all__ = [
    "FLServer",
    "EnsembleServer",
    "get_server",
    "fedavg",
    "fedmedian",
    "weighted_fedavg",
]
