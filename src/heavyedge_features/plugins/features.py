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
            help="csv file path of probabilistic classification labels.",
        )
        parser.add_config_argument(
            "--target-indices",
            type=int,
            nargs="+",
            help="List of indices of admissible classes.",
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
