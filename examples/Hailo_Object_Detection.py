from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    inference = ModelsZoo.load_yolov8ndet()

    for img in source:
        results = inference.predict(img)
        for result in results:
            print(result)
            # print(result.get_class_ids())
            # print(result.get_scores())
            # print(result.get_boxe_xyxy())
