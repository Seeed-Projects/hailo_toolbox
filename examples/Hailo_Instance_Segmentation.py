from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
from hailo_toolbox.process.visualization import SegmentationVisualization
import numpy as np
import cv2


def visualize_mask(mask):
    mask = mask.astype(np.uint8) * 155
    print(mask.sum())
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def visualize_masks(masks):
    for mask in masks:
        mask = visualize_mask(mask)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)


def visualize_boxes(img, boxes, labels):
    for box, label in zip(boxes, labels):
        cv2.rectangle(
            img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
        )
        cv2.putText(
            img,
            str(label),
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
    return img


if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load YOLOv8m segmentation model
    inference = ModelsZoo.segmentation.yolov8s_seg()

    visualization = SegmentationVisualization()

    for img in source:
        results = inference.predict(img)
        print(len(results))
        for result in results:
            labels = result.get_class_ids()
            if hasattr(result, "masks") and result.masks is not None:
                print(f"Segmentation result: mask shape {result.masks.shape}")
            if hasattr(result, "get_boxes_xyxy"):
                boxes = result.get_boxes_xyxy()
                print(f"Number of bounding boxes: {boxes}")
            if hasattr(result, "get_scores"):
                scores = result.get_scores()
                print(
                    f"Confidence scores: {scores[:5] if len(scores) > 5 else scores}"
                )  # Show first 5
            img = visualization.visualize(img, result)
            cv2.imshow("Segmentation", img)
            cv2.waitKey(1)
