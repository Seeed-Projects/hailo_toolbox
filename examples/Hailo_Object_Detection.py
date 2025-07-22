from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import DetectionVisualization
import cv2

if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load YOLOv8n detection model
    inference = ModelsZoo.detection.yolov8s()
    visualization = DetectionVisualization()
    for img in source:
        results = inference.predict(img)
        for result in results:
            img = visualization.visualize(img, result)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)
            # print(f"Detected {len(result)} objects")
            boxes = result.get_boxes()
            scores = result.get_scores()
            class_ids = result.get_class_ids()

            # Show first 5 detection results
            for i in range(min(5, len(result))):
                print(
                    f"  Object{i}: bbox{boxes[i]}, score{scores[i]:.3f}, class{class_ids[i]}"
                )
            print("---")
