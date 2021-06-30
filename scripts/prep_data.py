from hrl.prep.data_prepper import DataPrepper
from hrl.utils import setup_logger

logger = setup_logger("DEBUG")

import argparse

prepper = DataPrepper(
        tag = "events_29Jun2021",
        inputs = "../hrl/prep/metadata/samples_and_scale1fb.json",
        options = "../hrl/prep/metadata/options_default.json"
)

prepper.run()

