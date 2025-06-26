from hailo_toolbox.inference.core import InferenceEngine, CALLBACK_REGISTRY
from hailo_toolbox.process.preprocessor.preprocessor import PreprocessConfig
import yaml
import numpy as np
import cv2


@CALLBACK_REGISTRY.registryPostProcessor("custom")
class CustomPostProcessor:
    def __init__(self, config):
        self.config = config
        self.get_classes()

    def get_classes(self):
        with open("examples/ImageNet.yaml", "r") as f:
            self.classes = yaml.load(f, Loader=yaml.FullLoader)

    def __call__(self, results, original_shape=None):
        class_name = []
        for k, v in results.items():
            class_name.append(self.classes[np.argmax(v)])
        return class_name


@CALLBACK_REGISTRY.registryVisualizer("custom")
class CustomVisualizer:
    def __init__(self, config):
        self.config = config

    def __call__(self, original_frame, results):

        for v in results:
            cv2.putText(
                original_frame,
                f"CLASS: {v}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        return original_frame


@CALLBACK_REGISTRY.registryEventProcessor("custom")
class CustomEventProcessor:
    def __init__(self, config=None):
        self.config = config

    def __call__(self, results):
        print("Class Name:", results)
        return results


if __name__ == "__main__":
    preprocess_config = PreprocessConfig(
        target_size=(224, 224),
    )

    engine = InferenceEngine(
        model="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/efficientnet_s.hef",
        source="path/to/your/input/source",  # Replace with: image file, video file, folder path, or camera ID (0, 1, etc.)
        preprocess_config=preprocess_config,
        task_name="custom",
        show=True,
    )
    engine.run()
