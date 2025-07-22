from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2


def visualize_landmarks(img, landmarks_points):
    for x, y in landmarks_points:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return img


if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load hand landmark detection model
    inference = ModelsZoo.hand_landmark_detection.hand_landmark()

    for img in source:
        results = inference.predict(img)
        for result in results:
            landmarks = result.get_landmarks()
            landmarks_points = result.get_landmarks_points_with_original_shape()
            img = visualize_landmarks(img, landmarks_points)
            cv2.imshow("Hand Landmark Detection", img)
            cv2.waitKey(1)
            print(f"Hand Landmark Detection Result:")
            print(f"  Landmarks shape: {landmarks.shape}")
            print(f"  Number of landmarks: {len(landmarks)}")
            print(
                f"  Coordinate range: x[{landmarks.min():.1f}, {landmarks.max():.1f}], y[{landmarks.min():.1f}, {landmarks.max():.1f}]"
            )
            print(f"  Landmarks points: {landmarks_points}")

            print("---")
