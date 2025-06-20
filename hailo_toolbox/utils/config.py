from typing import Dict, Any, AnyStr, Optional
from dataclasses import dataclass


class ConvertConfig:
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
        return repr(self.__dict__)


class PreprocessConfig:
    pass


class PostprocessConfig:
    pass


class VisualizationConfig:
    pass


class CallbackConfig:
    pass


class InferConfig:
    model: AnyStr = None
    source: Any = None
    output: Any = None
    callback: Any = None
    preprocess: Any = None
    postprocess: Any = None
    visualization: Any = None

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
        return repr(self.__dict__)


@dataclass
class Config:
    command: AnyStr = None
    model: AnyStr = None
    callback: str = "base"
    convert: bool = False
    infer: bool = False
    source: Any = None
    output: Any = None
    preprocess: Any = None
    postprocess: Any = None
    visualization: Any = None
    task_type: Any = None
    save: bool = False
    save_path: Any = None
    show: bool = False

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
        return repr(self.__dict__)


if __name__ == "__main__":
    config = Config(
        {
            "model_path": "model.pt",
            "convert": True,
            "infer": True,
            "source": "video.mp4",
            "output": "output.mp4",
            "callback": "callback.py",
            "preprocess": "preprocess.py",
            "postprocess": "postprocess.py",
            "visualization": "visualization.py",
        }
    )
    print(repr(config))
