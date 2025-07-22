from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    source = create_source(
        "https://hailo-csdata.s3.eu-west-2.amazonaws.com/resources/video/example.mp4"
    )

    # Load CLIP ViT-L text-image retrieval model
    inference = ModelsZoo.text_image_retrieval.clip_vitb_16()

    for img in source:
        results = inference.predict(img)
        for result in results:
            print(f"Text-Image Retrieval Result:")
            print(f"  Result type: {type(result)}")
            # CLIP model typically returns image feature vectors for similarity calculation with text features
            if hasattr(result, "shape"):
                print(f"  Feature vector shape: {result.shape}")
            if hasattr(result, "get_embeddings"):
                embeddings = result.get_embeddings()
                print(f"  Feature vector dimension: {embeddings.shape[-1]}")
            print("---")
