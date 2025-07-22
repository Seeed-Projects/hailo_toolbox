from hailo_toolbox import create_source
from hailo_toolbox.models import ModelsZoo

if __name__ == "__main__":
    source = create_source(
        "rtsp://admin:pass8679@192.168.66.28:554/Streaming/Channels/1"
    )

    # Load LPRNet license plate recognition model
    inference = ModelsZoo.license_plate_recognition.lprnet()

    for img in source:
        results = inference.predict(img)
        for result in results:
            print(f"License Plate Recognition Result:")
            print(f"  Result type: {type(result)}")
            # Specific methods depend on actual result class implementation
            if hasattr(result, "text"):
                print(f"  Recognized text: {result.text}")
            if hasattr(result, "confidence"):
                print(f"  Confidence: {result.confidence}")
            print("---")
