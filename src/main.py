import argparse
import sys


class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Badura: An Algorithmic Composer supporting Difficulty Specification",
            usage='''badura <command> [<args>]''')

        parser.add_argument("command", help="Subcommand to run")

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, "command_" + args.command):
            print("Unknown command")
            parser.print_help()
            exit(1)

        parser = getattr(self, "command_" + args.command)()
        args = parser.parse_args(sys.argv[2:])

    @staticmethod
    def command_train():
        parser = argparse.ArgumentParser(description="Starts the training process")

        parser.add_argument("mode", choices=["preprocess", "lead", "acmp"],
                            help="Whether to load the training data, or start the training process of a network")

        return parser

    @staticmethod
    def command_gen():
        parser = argparse.ArgumentParser(description="Generates new MIDI files")

        parser.add_argument("--difficulty", "-d", type=int, required=True,
                            help="Desired difficulty of the piece")

        parser.add_argument("--primer", "-p", type=str, required=False,
                            help="Path to a primer MIDI file")

        return parser


if __name__ == '__main__':
    Parser()
