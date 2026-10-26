"""
Base class for all tools.
Every tool must inherit from BaseTool and implement the `call` method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class BaseTool(ABC):
    """Abstract base class that every tool must inherit from."""

    name: str = ""
    description: str = ""
    parameters: dict = {}

    def __init__(self, config: Dict = None):
        """
        Initialize the tool.

        Args:
            config: Tool configuration dictionary.
        """
        self.config = config or {}
        if not self.name:
            raise ValueError(f"Tool must have a name")

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        """
        Execute the tool.

        Args:
            params: Tool parameters.
            **kwargs: Additional keyword arguments (e.g., images, context).

        Returns:
            Tool execution result.
        """
        raise NotImplementedError

    def validate_params(self, params: dict) -> bool:
        """
        Validate whether the given parameters satisfy requirements.

        Args:
            params: Parameter dictionary.

        Returns:
            Whether validation passed.
        """
        required = self.parameters.get("required", [])
        return all(key in params for key in required)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
