"""Commands for edge feature quantification."""

import pathlib

from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 2.0


@register_command("features-global", "Quantify global shape features")
class GlobalFeaturesCommand(Command):
    def add_parser(self, main_parser):
        parser = main_parser.add_parser(
            self.name,
            help="Quantify global shape feature score using classification model",
            epilog=(
                "The input label can be in npy (default) or csv format. "
                "If csv, the first row is the header. "
                "Unrecognized formats are parsed as npy with a warning. "
                "Output field 'phi' is the signed distance to the "
                "nearest admissible class in the probability simplex."
            ),
        )
        parser.add_argument(
            "soft_labels",
            type=pathlib.Path,
            help="Path to file containing probabilistic classification labels.",
        )
        parser.add_config_argument(
            "--target-indices",
            type=int,
            nargs="+",
            help="List of indices of admissible classes from trained labels.",
        )
        parser.add_argument(
            "--label-format",
            choices=["npy", "csv"],
            help="Label file format. If not passed, parsed from file extension.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=pathlib.Path,
            help="Path to output csv file",
        )

    def run(self, args):
        import csv
        import os

        import numpy as np

        from heavyedge_features.api import global_deviation

        if args.target_indices is None:
            raise ValueError("--target-indices must be specified.")

        label_ext = os.path.splitext(args.soft_labels)[1].lower().lstrip(".")
        label_format = args.label_format or label_ext

        self.logger.info(f"Start processing {args.output}")

        if label_format == "csv":
            with open(args.soft_labels, "r") as f:
                reader = csv.reader(f)
                # Burn first row as header
                next(reader)
                soft_labels = np.array([row[0] for row in reader])
        else:
            if label_format != "npy":
                self.logger.warning(
                    f"Unrecognized label format '{label_format}', parsing as npy."
                )
            soft_labels = np.load(args.soft_labels)

        values = global_deviation(
            soft_labels,
            args.target_indices,
            logger=lambda msg: self.logger.info(f"{args.output} : {msg}"),
        )

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["phi"])
            for value in values:
                writer.writerow([value])

        self.logger.info(f"Saved {args.output}.")


@register_command("features-local", "Quantify local shape features")
class LocalFeaturesCommand(Command):
    def add_parser(self, main_parser):
        parser = main_parser.add_parser(
            self.name,
            help="Quantify local shape features",
            epilog=(
                "The input label can be in npy (default) or csv format. "
                "If csv, the first row is the header. "
                "Unrecognized formats are parsed as npy with a warning. "
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
            "soft_labels",
            type=pathlib.Path,
            help="Path to file containing probabilistic classification labels.",
        )
        parser.add_argument(
            "h_w",
            type=pathlib.Path,
            help="Path to npy file containing wet thickness.",
        )
        parser.add_config_argument(
            "--sigma",
            type=float,
            help="Standard deviation of Gaussian kernel.",
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
            "--label-format",
            choices=["npy", "csv"],
            help="Label file format. If not passed, parsed from file extension.",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=pathlib.Path,
            help="Path to output csv file",
        )

    def run(self, args):
        import csv
        import os

        import numpy as np
        from heavyedge import ProfileData

        from heavyedge_features.api import edge_height, edge_width

        if args.type1_indices is None:
            raise ValueError("--type1-indices must be specified.")
        if args.type2_indices is None:
            raise ValueError("--type2-indices must be specified.")
        if args.type3_indices is None:
            raise ValueError("--type3-indices must be specified.")

        label_ext = os.path.splitext(args.soft_labels)[1].lower().lstrip(".")
        label_format = args.label_format or label_ext

        self.logger.info(f"Start processing {args.output}")

        if label_format == "csv":
            with open(args.soft_labels, "r") as f:
                reader = csv.reader(f)
                # Burn first row as header
                next(reader)
                soft_labels = np.array([row[0] for row in reader])
        else:
            if label_format != "npy":
                self.logger.warning(
                    f"Unrecognized label format '{label_format}', parsing as npy."
                )
            soft_labels = np.load(args.soft_labels)

        wet_thicknesses = np.load(args.h_w)

        edge_heights = edge_height(
            ProfileData(args.profiles),
            logger=lambda msg: self.logger.info(f"{args.output} : {msg}"),
        )
        edge_widths = edge_width(
            ProfileData(args.profiles),
            soft_labels.argmax(axis=1),
            wet_thicknesses,
            args.sigma,
            args.type1_indices,
            args.type2_indices,
            args.type3_indices,
            logger=lambda msg: self.logger.info(f"{args.output} : {msg}"),
        )

        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["H", "b"])
            for H, b in zip(edge_heights, edge_widths):
                writer.writerow([H, b])

        self.logger.info(f"Saved {args.output}.")
