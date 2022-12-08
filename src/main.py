import argparse

from sCoda import Sequence, Bar

from src.config.settings import DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH, DATA_TRAIN_OUTPUT_FILE_PATH, \
    DATA_BARS_VAL_OUTPUT_FOLDER_PATH, DATA_VAL_OUTPUT_FILE_PATH, DATA_MIDI_INPUT_PATH, PATH_MIDI
from src.network.paul import train_network, generate, store_checkpoint
from src.preprocessing.preprocessing import store_records, load_midi, clean_midi
from src.util.enumerations import NetworkType
from src.util.logging import get_logger
from src.util.util import get_prj_root


def main():
    args, parser = parse_arguments()
    logger = get_logger(__name__)

    if not hasattr(args, "command"):
        parser.print_help()
        return

    if args.command == "preprocess":
        logger.info("Preprocessing MIDI files...")

        if args.clean:
            logger.info("Cleaning MIDI files...")

            clean_midi(DATA_MIDI_INPUT_PATH)
            return

        if args.skip_midi:
            logger.info("Skipping loading of MIDI files...")
        else:
            logger.info("Loading MIDI files...")
            load_midi(DATA_MIDI_INPUT_PATH)

        logger.info("Storing TFRecords...")
        store_records(input_dir=DATA_BARS_TRAIN_OUTPUT_FOLDER_PATH, output_dir=DATA_TRAIN_OUTPUT_FILE_PATH)
        store_records(input_dir=DATA_BARS_VAL_OUTPUT_FOLDER_PATH, output_dir=DATA_VAL_OUTPUT_FILE_PATH)

        logger.info("Successfully"
                    " processed MIDI files.")
    elif args.command == "train":
        run_identifier = args.run_identifier

        if args.network == "lead":
            logger.info("Starting training process for lead network...")

            train_network(network_type=NetworkType.lead, run_identifier=run_identifier)

            logger.info("Successfully trained lead network.")
        elif args.network == "acmp":
            logger.info("Starting training process for acmp network...")

            train_network(network_type=NetworkType.acmp, run_identifier=run_identifier)

            logger.info("Successfully trained acmp network.")
    elif args.command == "store":
        network_type = NetworkType.lead if args.network == "lead" else NetworkType.acmp
        store_checkpoint(network_type=network_type, run_identifier=args.run_identifier,
                         checkpoint_identifier=int(args.checkpoint_identifier))
    elif args.command == "generate":
        times = 1
        if args.times is not None:
            times = args.times

        for _ in range(times):
            if args.track == "lead":
                logger.info(f"Generating lead track with difficulty {args.difficulty}...")

                generate(network_type=NetworkType.lead, model_identifier=args.model_identifier,
                         difficulty=args.difficulty - 1)
            elif args.track == "acmp":
                logger.info(f"Generating acmp track with difficulty {args.difficulty}...")

                folder = NetworkType.lead.value
                if args.input_folder is not None:
                    folder = args.input_folder

                sequence_path = f"{PATH_MIDI}/{folder}/{args.seq_identifier}.mid"
                sequence_lead = Sequence.from_midi_file(sequence_path, [[0]], [0])[0]

                generate(network_type=NetworkType.acmp, model_identifier=args.model_identifier,
                         difficulty=args.difficulty - 1, lead_seq=sequence_lead, name=args.seq_identifier)


def parse_arguments():
    parser = argparse.ArgumentParser(description="PAUL-2: An Algorithmic Composer supporting Difficulty Specification")
    subparsers = parser.add_subparsers(title="Valid Commands",
                                       help="Selects the mode of operation.")

    # Preprocess Command
    parser_train = subparsers.add_parser("preprocess", aliases=["p"], help="Preprocesses data.")
    parser_train.set_defaults(command="preprocess")
    parser_train.add_argument("--skip_midi", action="store_true",
                              help="Skips processing and storing of MIDI files.")
    parser_train.add_argument("--clean", action="store_true",
                              help="Removes all invalid files from the dataset.")

    # Train Command
    parser_train = subparsers.add_parser("train", aliases=["t"], help="Trains the networks.")
    parser_train.set_defaults(command="train")
    parser_train.add_argument("network", choices=["lead", "acmp"],
                              help="Selects which network to train.")
    parser_train.add_argument("--run_identifier", "-r",
                              help="Determines which run to continue.")

    # Store Command
    parser_generate = subparsers.add_parser("store", aliases=["s"],
                                            help="Loads a given checkpoint and stores it as h5 model.")
    parser_generate.set_defaults(command="store")
    parser_generate.add_argument("network", choices=["lead", "acmp"],
                                 help="Selects which network to store.")
    parser_generate.add_argument("--run_identifier", "-r", required=True,
                                 help="Determines which run to store.")
    parser_generate.add_argument("--checkpoint_identifier", "-c", required=True,
                                 help="Determines which checkpoint to store.")

    # Generate Command
    parser_generate = subparsers.add_parser("generate", aliases=["g"],
                                            help="Generates new music based on the given parameters.")
    parser_generate.set_defaults(command="generate")
    parser_generate.add_argument("track", choices=["lead", "acmp", "combined"],
                                 help="Selects which track to generate.")
    parser_generate.add_argument("--difficulty", "-d", type=int, choices=range(1, 11), required=True,
                                 help="Determines the difficulty of the tracks to generate.")
    parser_generate.add_argument("--model_identifier", "-i", required=True,
                                 help="Determines which model to load.")
    parser_generate.add_argument("--seq_identifier", "-s", required=False,
                                 help="Determines which sequence to load for the lead track.")
    parser_generate.add_argument("--times", "-t", type=int, choices=range(1, 11), required=False,
                                 help="Determines how often to run the procedure.")
    parser_generate.add_argument("--input_folder", "-f", required=False,
                                 help="Determines which folder to load the leading samples from.")

    args = parser.parse_args()
    return args, parser


if __name__ == '__main__':
    main()
