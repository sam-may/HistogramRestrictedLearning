import argparse

from hrl.utils import setup_logger
from hrl.prep.scale1fb import Scale1fbHelper

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    help = "path to input json",
    type = str,
    default = "../hrl/prep/metadata/samples.json"
)
parser.add_argument(
    "--output",
    help = "path for output json",
    type = str,
    default = "../hrl/prep/metadata/samples_and_scale1fb.json"
)
args = parser.parse_args()

logger = setup_logger("DEBUG", "output/scale1fb_log.txt")
logger.info("[calculate_scale1fb.py] Running calculate_scale1fb.py with arguments: %s" % str(args))

scale1fb_helper = Scale1fbHelper(
        input = args.input,
        output_file = args.output,
        logger = logger
)

scale1fb_helper.run() 

