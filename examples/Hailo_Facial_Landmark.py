from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2


def visualize_facial_landmark(img, landmarks):
    for i in range(len(landmarks)):
        cv2.circle(
            img, (int(landmarks[i][0]), int(landmarks[i][1])), 3, (0, 0, 255), -1
        )
    return img


if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load TDDFA facial landmark detection model
    inference = ModelsZoo.facial_landmark.tddfa()

    for img in source:
        results = inference.predict(img)
        for result in results:
            landmarks = result.get_landmarks_with_original_shape()
            original_shape = result.get_original_shape()

            print(f"Facial Landmark Detection Result:")
            print(f"  Landmarks shape: {landmarks.shape}")
            print(f"  Number of landmarks: {len(landmarks)} (standard 68 points)")
            print(f"  Original image size: {original_shape}")
            print(
                f"  Landmark coordinate range: x[{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}], y[{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]"
            )
            img = visualize_facial_landmark(img, landmarks)
            cv2.imshow("Facial Landmark Detection", img)
            cv2.waitKey(1)
            # Landmark distribution statistics
            x_std = landmarks[:, 0].std()
            y_std = landmarks[:, 1].std()
            print(f"  Landmark distribution: x_std={x_std:.3f}, y_std={y_std:.3f}")
            print("---")
