from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo
import numpy as np

if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load OSNet-X1 person re-identification model
    inference = ModelsZoo.person_reid.osnet_x1()

    for img in source:
        results = inference.predict(img)
        for result in results:
            embeddings = result.get_embeddings()
            original_shape = result.get_original_shape()

            print(f"Person Re-identification Result:")
            print(f"  Feature vector shape: {embeddings.shape}")
            print(f"  Feature vector dimension: {embeddings.shape[-1]}")
            print(f"  Original image size: {original_shape}")
            print(f"  Feature vector norm: {np.linalg.norm(embeddings):.3f}")
            print(
                f"  Feature vector range: [{embeddings.min():.3f}, {embeddings.max():.3f}]"
            )
            print("---")
