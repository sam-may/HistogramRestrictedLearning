import argparse
import json

from hrl.utils.logger_utils import setup_logger
from hrl.algorithms.dnn_helper import DNNHelper

def parse_arguments():
    parser = argparse.ArgumentParser(
            description = "Train DNNs, possibly with domain adaptation component, and save results.")

    # Required arguments
    parser.add_argument(
            "--input_dir_cl",
            required=True,
            type=str,
            help="path to input directory for classification events with `.parquet` file produced by HiggsDNA") 

    parser.add_argument(
            "--output_dir",
            required=True,
            type=str,
            help="path to output directory")

    # Optional arguments
    parser.add_argument(
            "--input_dir_da",
            required=False,
            default=None,
            type=str,
            help="path to input directory for domain adaptation events with `.parquet` file produced by HiggsDNA")

    parser.add_argument(
            "--max_events",
            required=False,
            default=-1,
            type=int,
            help="maximum number of events to load from `.parquet` files")

    parser.add_argument(
        "--log-level",
        required=False,
        default="DEBUG",
        type=str,
        help="Level of information printed by the logger") 

    parser.add_argument(
        "--log-file",
        required=False,
        type=str,
        help="Name of the log file")

    return parser.parse_args()


def main(args):
    logger = setup_logger(args.log_level, args.log_file)
    
    dnn_helper = DNNHelper(**vars(args)) 
    dnn_helper.run()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
