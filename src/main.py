import argparse

from src.util.logging import get_logger


def main():
    args = parse_arguments()
    logger = get_logger(__name__)

    if args.command == "train":
        if args.train_mode == "preprocess":
            logger.info("Preprocessing MIDI files.")

            # load_midi_files(DATA_MIDI_INPUT_PATH)
            logger.info("Successfully processed MIDI files.")
    elif args.command == "generate":
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Badura: An Algorithmic Composer supporting Difficulty Specification")
    subparsers = parser.add_subparsers(title="Valid Commands",
                                       help="Selects the mode of operation.")

    # Train Command
    parser_train = subparsers.add_parser("train", aliases=["t"], help="Trains the networks or preprocesses data.")
    parser_train.set_defaults(command="train")
    parser_train.add_argument("train_mode", choices=["preprocess"], help="Which option of the train suite to run.")

    # Generate Command
    parser_generate = subparsers.add_parser("generate", aliases=["g"],
                                            help="Generates new music based on the given parameters.")
    parser_generate.set_defaults(command="generate")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
