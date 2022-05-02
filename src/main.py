import argparse

from src.network.badura import train_network, NetworkType
from src.preprocessing.data_pipeline import load_and_store_records, load_midi_files
from src.config.settings import DATA_MIDI_INPUT_PATH
from src.util.logging import get_logger


def main():
    args = parse_arguments()
    logger = get_logger(__name__)

    if args.command == "preprocess":
        logger.info("Preprocessing MIDI files...")

        if args.skip_midi:
            logger.info("Skipping loading of MIDI files...")
        else:
            logger.info("Loading MIDI files...")
            load_midi_files(DATA_MIDI_INPUT_PATH)

        logger.info("Storing TFRecords...")
        load_and_store_records()

        logger.info("Successfully processed MIDI files.")
    elif args.command == "train":
        if args.network == "lead":
            logger.info("Starting training process for lead network...")

            train_network(network_type=NetworkType.lead)

            logger.info("Successfully trained lead network.")
        elif args.network == "acmp":
            logger.info("Starting training process for acmp network...")

            train_network(network_type=NetworkType.acmp)

            logger.info("Successfully trained acmp network.")
    elif args.command == "generate":
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Badura: An Algorithmic Composer supporting Difficulty Specification")
    subparsers = parser.add_subparsers(title="Valid Commands",
                                       help="Selects the mode of operation.")

    # Preprocess Command
    parser_train = subparsers.add_parser("preprocess", aliases=["p"], help="Preprocesses data.")
    parser_train.set_defaults(command="preprocess")
    parser_train.add_argument("--skip-midi", action="store_true",
                              help="Skips processing and storing of MIDI files.")

    # Train Command
    parser_train = subparsers.add_parser("train", aliases=["t"], help="Trains the networks.")
    parser_train.set_defaults(command="train")
    parser_train.add_argument("network", choices=["lead", "acmp"],
                              help="Selects which network to train.")

    # Generate Command
    parser_generate = subparsers.add_parser("generate", aliases=["g"],
                                            help="Generates new music based on the given parameters.")
    parser_generate.set_defaults(command="generate")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
