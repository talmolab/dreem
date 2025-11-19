"""Base processor classes for DREEM processing pipelines."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ProcessingStep(ABC):
    """Abstract base class for processing steps.

    Each step accepts a state dict, modifies it, and returns it.
    Steps are instantiated once and can be called multiple times in a loop.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize a processing step.

        Args:
            name: Optional name for the step. Defaults to class name.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the processing step.

        Args:
            state: Dictionary containing all necessary data for processing.
                   Modified in place and returned.

        Returns:
            The modified state dictionary.
        """
        ...
