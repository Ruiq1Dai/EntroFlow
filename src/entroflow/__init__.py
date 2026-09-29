"""EntroFlow runtime communication-entropy plug-in core."""

from .entropy import EdgeEntropyEvent, RollingEdgeEntropy, normalized_entropy
from .execution import ActivatedGraph, ExecutionInstance
from .topology import StaticTopology

__all__ = [
    "ActivatedGraph",
    "EdgeEntropyEvent",
    "ExecutionInstance",
    "RollingEdgeEntropy",
    "StaticTopology",
    "normalized_entropy",
]
__version__ = "0.2.0"
