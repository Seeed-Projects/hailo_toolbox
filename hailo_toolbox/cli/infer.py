"""
Command-line interface for model inference.
"""

import hailo_toolbox as ht
from hailo_toolbox.cli import parse_args
from hailo_toolbox.utils.config import Config
from hailo_toolbox.inference import InferenceEngine


def main():
    args = parse_args()
    config = Config(vars(args))
    if args.command == "infer":
        engine = InferenceEngine(**vars(args))
        engine.run()
    elif args.command == "convert":
        raise NotImplementedError("Conversion is not implemented yet")


if __name__ == "__main__":
    main()
