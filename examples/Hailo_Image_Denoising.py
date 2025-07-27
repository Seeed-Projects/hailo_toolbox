from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import cv2


if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load DnCNN3 image denoising model
    inference = ModelsZoo.image_denoise.dncnn3()

    for img in source:
        results = inference.predict(img)
        for result in results:
            denoised_image = result.get_denoised_image()
            original_shape = result.get_original_shape()
            cv2.imshow("Denoised Image", denoised_image[..., ::-1])
            cv2.waitKey(1)

            print(f"Image Denoising Result:")
            print(f"  Original image size: {original_shape}")
            print(f"  Denoised image shape: {denoised_image.shape}")
            print(f"  Original image mean brightness: {img.mean():.1f}")
            print(f"  Denoised image mean brightness: {denoised_image.mean():.1f}")
            print("---")
