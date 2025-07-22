from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import KeypointVisualization
import cv2

if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load YOLOv8s pose estimation model
    inference = ModelsZoo.pose_estimation.yolov8s_pose()
    visualization = KeypointVisualization()
    for img in source:
        results = inference.predict(img)
        for result in results:
            img = visualization.visualize(img, result)
            cv2.imshow("Pose Estimation", img)
            cv2.waitKey(1)
            print(f"Detected {len(result)} persons")
            # Show pose information for first 3 persons
            for i, person in enumerate(result):
                if i >= 3:  # Only show first 3
                    break
                keypoints = person.get_keypoints()
                score = person.get_score()
                boxes = person.get_boxes()
                print(keypoints.shape, score.shape, boxes.shape)
                print(
                    f"  Person{i}: {len(keypoints)} keypoints, confidence{score[0]:.3f}, bbox{boxes}"
                )
            print("---")
