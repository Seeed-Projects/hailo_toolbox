from typing import Dict, Any, AnyStr, Optional
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration class for backward compatibility.

    This class maintains compatibility with existing code while
    the InferenceEngine now accepts individual parameters directly.
    """

    command: Optional[AnyStr] = None
    model: Optional[AnyStr] = None
    task_name: str = "yolov8det"
    convert: bool = False
    infer: bool = False
    source: Any = None
    output: Any = None
    preprocess_config: Any = None
    postprocess_config: Any = None
    visualization_config: Any = None
    task_type: Any = None
    save: bool = False
    save_path: Any = None
    show: bool = False

    def __init__(self, config: Optional[Dict[AnyStr, Any]] = None):
        """
        Initialize configuration with optional dictionary.

        Args:
            config: Optional dictionary containing configuration parameters
        """
        if config is not None:
            self.update(config)

    def update(self, config: Dict[AnyStr, Any]) -> None:
        """
        Update configuration with dictionary values.

        Args:
            config: Dictionary containing configuration parameters to update
        """
        self.__dict__.update(config)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format suitable for InferenceEngine.

        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }

    def __getitem__(self, key: AnyStr) -> Any:
        """Get configuration value by key."""
        return self.__dict__[key]

    def __setitem__(self, key: AnyStr, value: Any) -> None:
        """Set configuration value by key."""
        self.__dict__[key] = value

    def __delitem__(self, key: AnyStr) -> None:
        """Delete configuration value by key."""
        del self.__dict__[key]

    def __str__(self) -> str:
        """
        Return a string representation of the Config object.

        Returns:
            A string representation of the Config object.
        """
        result = ""
        for key, value in self.__dict__.items():
            result += f"{key.center(20)}: {value}\n"
        return result

    def __repr__(self) -> str:
        """Return a detailed representation of the Config object."""
        return repr(self.__dict__)


# Legacy classes for backward compatibility
class ConvertConfig:
    """Legacy class for conversion configuration."""

    model: AnyStr = None
    source: Any = None

    def __init__(self, config: Dict[AnyStr, Any]):
        if config is not None:
            self.update(config)

    def update(self, config: Dict[AnyStr, Any]):
        self.__dict__.update(config)

    def __getitem__(self, key: AnyStr) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: AnyStr, value: Any):
        self.__dict__[key] = value

    def __delitem__(self, key: AnyStr):
        del self.__dict__[key]

    def __str__(self) -> str:
        result = ""
        for key, value in self.__dict__.items():
            result += f"{key.center(20)}: {value}\n"
        return result

    def __repr__(self) -> str:
        return repr(self.__dict__)


class PreprocessConfig:
    """Legacy class for preprocessing configuration."""

    pass


class PostprocessConfig:
    """Legacy class for postprocessing configuration."""

    pass


class VisualizationConfig:
    """Legacy class for visualization configuration."""

    pass


class CallbackConfig:
    """Legacy class for callback configuration."""

    pass


class InferConfig:
    """Legacy class for inference configuration."""

    model: AnyStr = None
    source: Any = None
    output: Any = None
    task_name: Any = "yolov8det"
    preprocess_config: Any = None
    postprocess_config: Any = None
    visualization_config: Any = None

    def __init__(self, config: Dict[AnyStr, Any]):
        if config is not None:
            self.update(config)

    def update(self, config: Dict[AnyStr, Any]):
        self.__dict__.update(config)

    def __getitem__(self, key: AnyStr) -> Any:
        return self.__dict__[key]

    def __setitem__(self, key: AnyStr, value: Any):
        self.__dict__[key] = value

    def __delitem__(self, key: AnyStr):
        del self.__dict__[key]

    def __str__(self) -> str:
        result = ""
        for key, value in self.__dict__.items():
            result += f"{key.center(20)}: {value}\n"
        return result

    def __repr__(self) -> str:
        return repr(self.__dict__)


if __name__ == "__main__":
    # Example usage demonstrating backward compatibility
    config = Config(
        {
            "model": "model.pt",
            "convert": True,
            "infer": True,
            "source": "video.mp4",
            "output": "output.mp4",
            "task_name": "yolov8det",
            "preprocess_config": "preprocess.py",
            "postprocess_config": "postprocess.py",
            "visualization_config": "visualization.py",
        }
    )
    print("Config object:", repr(config))
    print("Config as dict:", config.to_dict())
