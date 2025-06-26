from hailo_toolbox.inference import InferenceEngine


if __name__ == "__main__":
    engine = InferenceEngine(
        model="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef",
        source="Enter the path to your input source",
        task_name="yolov8pe",
        show=True,
    )
    engine.run()
