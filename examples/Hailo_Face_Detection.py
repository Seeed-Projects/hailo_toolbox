from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2


def visualize_face_detection(img, boxes, scores, landmarks):
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        print(
            f"  Face{i}: bbox[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}], confidence{score:.3f}"
        )
        cv2.rectangle(
            img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2
        )

    return img


if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load SCRFD-10G face detection model
    inference = ModelsZoo.face_detection.retinaface_mbnet()

    for img in source:
        results = inference.predict(img)
        for result in results:
            print(f"Detected {len(result)} faces")

            boxes = result.get_boxes(pixel_coords=True)
            scores = result.get_scores()
            landmarks = result.get_landmarks(pixel_coords=True)
            print(boxes.shape, scores.shape)
            img = visualize_face_detection(img, boxes, scores, landmarks)
            cv2.imshow("Face Detection", img)
            cv2.waitKey(1)

            # Show first 3 face detection results
            for i in range(min(3, len(result))):
                box = boxes[i]
                score = scores[i]
                print(
                    f"  Face{i}: bbox[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}], confidence{score:.3f}"
                )
                if landmarks is not None:
                    face_landmarks = landmarks[i]
                    print(f"    Landmarks: {len(face_landmarks)//2} points")
            print("---")
