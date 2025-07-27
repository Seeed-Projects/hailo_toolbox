from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2


if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load FastDepth depth estimation model
    inference = ModelsZoo.depth_estimation.fast_depth()

    for img in source:
        results = inference.predict(img)
        for result in results:
            depth_map = result.get_depth()
            depth_normalized = result.get_depth_normalized()
            original_shape = result.get_original_shape()
            print(depth_map)
            cv2.imshow("Depth Estimation", depth_map)
            cv2.waitKey(1)

            print(f"Depth Estimation Result:")
            print(f"  Depth map shape: {depth_map.shape}")
            print(
                f"  Depth value range: [{depth_map.min():.3f}, {depth_map.max():.3f}]"
            )
            print(f"  Original image size: {original_shape}")
            print("---")
