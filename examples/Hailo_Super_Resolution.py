from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load Real-ESRGAN super resolution model
    inference = ModelsZoo.super_resolution.real_esrgan()

    for img in source:
        results = inference.predict(img)
        for result in results:
            enhanced_image = result.get_image()
            original_shape = result.get_original_shape()
            input_shape = result.get_input_shape()
            cv2.imshow("Enhanced Image", enhanced_image)
            cv2.waitKey(1)
            print(f"Super Resolution Result:")
            print(f"  Original image size: {original_shape}")
            print(f"  Enhanced image shape: {enhanced_image.shape}")
            print(f"  Input image shape: {input_shape}")
            print(
                f"  Upscale factor: {enhanced_image.shape[0] / original_shape[0]:.1f}x"
            )
            print("---")
