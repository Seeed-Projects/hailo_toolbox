from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2

if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load Zero-DCE low light enhancement model
    inference = ModelsZoo.low_light_enhancement.zero_dce()

    for img in source:
        results = inference.predict(img // 10)
        for result in results:
            enhanced_image = result.get_enhanced_image()
            original_shape = result.get_original_shape()
            cv2.imshow("Enhanced Image", enhanced_image[..., ::-1])
            cv2.imshow("Original Image", img // 10)
            cv2.waitKey(1)

            print(f"Low Light Enhancement Result:")
            print(f"  Original image size: {original_shape}")
            print(f"  Enhanced image shape: {enhanced_image.shape}")
            print(f"  Original image mean brightness: {img.mean():.1f}")
            print(f"  Enhanced image mean brightness: {enhanced_image.mean():.1f}")
            print(
                f"  Brightness improvement: {enhanced_image.mean() / img.mean():.2f}x"
            )
            print("---")
