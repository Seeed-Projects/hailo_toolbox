from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import numpy as np

if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load R3D-18 video classification model
    inference = ModelsZoo.video_classification.r3d_18()

    # Collect multiple frames for video input (R3D requires multi-frame input)
    frames = []
    frame_count = 0

    for img in source:
        frames.append(img)
        frame_count += 1

        # Process every 16 frames (typical for R3D-18)
        if len(frames) == 16:
            # Stack frames to create video input
            video_input = np.stack(frames, axis=0)  # Shape: (16, H, W, 3)

            results = inference.predict(video_input)
            for result in results:
                class_names = result.get_class_name_top5()
                class_indices = result.get_class_index_top5()
                scores = result.get_score_top5()
                original_shape = result.get_original_shape()

                print(f"Video Classification Result:")
                print(f"  Original video size: {original_shape}")
                print(f"  Top5 classes: {class_names}")
                print(f"  Top5 indices: {class_indices}")
                print(f"  Top5 scores: {[f'{score:.3f}' for score in scores]}")
                print(
                    f"  Most likely action: {class_names[0]} (confidence: {scores[0]:.3f})"
                )
                print("---")

            # Clear frames for next batch
            frames = []

        # Process only first few batches for demo
        if frame_count >= 32:
            break
