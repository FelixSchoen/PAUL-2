import argparse

from src.data_processing.data_pipeline import load_and_store_records
from src.network.badura import train_lead
from src.util.logging import get_logger


def main():
    args = parse_arguments()
    logger = get_logger(__name__)

    if args.command == "train":
        if args.train_mode == "preprocess":
            logger.info("Preprocessing MIDI files...")

            logger.info("Loading MIDI files...")
            # load_midi_files(DATA_MIDI_INPUT_PATH)

            logger.info("Storing TFRecords...")
            load_and_store_records()

            logger.info("Successfully processed MIDI files.")
        elif args.train_mode == "lead":
            logger.info("Starting training process for lead network...")

            train_lead()

            logger.info("Successfully trained lead network.")
    elif args.command == "generate":
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Badura: An Algorithmic Composer supporting Difficulty Specification")
    subparsers = parser.add_subparsers(title="Valid Commands",
                                       help="Selects the mode of operation.")

    # Train Command
    parser_train = subparsers.add_parser("train", aliases=["t"], help="Trains the networks or preprocesses data.")
    parser_train.set_defaults(command="train")
    parser_train.add_argument("train_mode", choices=["preprocess", "lead"],
                              help="Which option of the train suite to run.")

    # Generate Command
    parser_generate = subparsers.add_parser("generate", aliases=["g"],
                                            help="Generates new music based on the given parameters.")
    parser_generate.set_defaults(command="generate")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
