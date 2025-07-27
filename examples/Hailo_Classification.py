from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load MobileNetV1 classification model
    inference = ModelsZoo.classification.resnet18()

    for img in source:
        results = inference.predict(img)
        for result in results:
            print(
                f"Classification result: {result.get_class_name()} (confidence: {result.get_score():.3f})"
            )
            print(f"Top5 classes: {result.get_top_5_class_names()}")
            print(
                f"Top5 scores: {[f'{score:.3f}' for score in result.get_top_5_scores()]}"
            )
            print("---")
