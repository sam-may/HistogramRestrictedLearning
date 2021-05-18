from hrl.prep.data_prepper import DataPrepper
from hrl.utils import setup_logger

import argparse

prepper = DataPrepper(
        tag = "test",
        inputs = "../hrl/prep/metadata/samples_and_scale1fb.json",
        options = "../hrl/prep/metadata/options_default.json"
)

prepper.run()

