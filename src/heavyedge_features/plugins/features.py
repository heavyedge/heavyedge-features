"""Commands for edge feature quantification."""

import pathlib

from heavyedge import ProfileData
from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 2.0


@register_command("features-global", "Quantify global shape features")
class GlobalFeaturesCommand(Command):
    def add_parser(self, main_parser):
        parser = main_parser.add_parser(
            self.name,
            help="Quantify global shape feature score using classification model",
            epilog=(
                "Output field 'phi' is the signed distance to the "
                "nearest admissible class in the probability simplex."
            ),
        )
        parser.add_argument(
            "profiles",
            type=pathlib.Path,
            help="h5 file path to profile data in 'ProfileData' structure.",
        )
        parser.add_argument(
            "model",
            type=pathlib.Path,
            help="Path to trained classification model.",
        )
        parser.add_config_argument(
            "--target-indices",
            type=int,
            nargs="+",
            help="List of indices of admissible classes from trained labels.",
        )
        parser.add_argument(
            "--normalized",
            action="store_true",
            help=(
                "If input profiles are already normalized, "
                "setting this flag enhances performance."
            ),
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help=(
                "Batch size to load data. "
                "If not passed, all data are loaded at once."
            ),
        )
        parser.add_argument(
            "-o",
            "--output",
            type=pathlib.Path,
            help="Path to output csv file",
        )

    def run(self, args):
        import csv
        import pickle

        from heavyedge_features.api import global_deviation

        if args.target_indices is None:
            raise ValueError("--target-indices must be specified.")

        self.logger.info(f"Start processing {args.output}")

        profiles = ProfileData(args.profiles)
        with open(args.model, "rb") as f:
            model = pickle.load(f)

        generator = global_deviation(
            profiles,
            args.target_indices,
            model,
            normalize=not args.normalized,
            batch_size=args.batch_size,
            logger=lambda msg: self.logger.info(f"{args.output} : {msg}"),
        )

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phi"])
            for batch in generator:
                for value in batch:
                    writer.writerow([value])

        self.logger.info(f"Saved {args.output}.")


@register_command("features-local", "Quantify local shape features")
class LocalFeaturesCommand(Command):
    def add_parser(self, main_parser):
        parser = main_parser.add_parser(
            self.name,
            help="Quantify local shape features",
            epilog=(
                "Output field 'H' is the apparent edge superelevation. "
                "Output field 'b' is the edge width."
            ),
        )
        parser.add_argument(
            "profiles",
            type=pathlib.Path,
            help="h5 file path to profile data in 'ProfileData' structure.",
        )
        parser.add_argument(
            "model",
            type=pathlib.Path,
            help="Path to trained classification model.",
        )
        parser.add_argument(
            "h_w",
            type=pathlib.Path,
            help="Path to npy file containing wet thickness.",
        )
        parser.add_config_argument(
            "--type1-indices",
            type=int,
            nargs="+",
            help="List of indices of Type 1 classes from trained labels.",
        )
        parser.add_config_argument(
            "--type2-indices",
            type=int,
            nargs="+",
            help="List of indices of Type 2 classes from trained labels.",
        )
        parser.add_config_argument(
            "--type3-indices",
            type=int,
            nargs="+",
            help="List of indices of Type 3 classes from trained labels.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help=(
                "Batch size to load data. "
                "If not passed, all data are loaded at once."
            ),
        )
        parser.add_argument(
            "-o",
            "--output",
            type=pathlib.Path,
            help="Path to output csv file",
        )

    def run(self, args):
        raise NotImplementedError
