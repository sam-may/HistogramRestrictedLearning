# HistogramRestrictedLearning 

## Installation

**1. Clone repository**
```
git clone https://github.com/sam-may/HistogramRestrictedLearning.git
cd HistogramRestrictedLearning
```
**2. Install dependencies**

Dependencies are listed in ```environment.yml```. Install with
```
conda env create -f environment.yml
```

**3. Install hrl**

**Users** can install with:
```
python setup.py install
```
**Developers** are suggested to install with:
```
pip install -e .
```
to avoid rerunning the whole installation every time there is a change.

Once your setup is installed, you can activate your python environment with
```
conda activate hrl
```

**Note**: `CMSSW` environments can interfere with `conda` environments. Recommended to unset your CMSSW environment (if any) by running
```
eval `scram unsetenv -sh`
```
before attempting installation and each time before activating the `conda` environment.

## Development Guidelines

### Documentation
Please comment code following [this convention](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) from `sphinx`.

In the future, `sphinx` can be used to automatically generate documentation pages for this project.

### Logging
Logging currently uses the Python [logging facility](https://docs.python.org/3/library/logging.html) together with [rich](https://github.com/willmcgugan/rich) (for pretty printing) to provide useful information printed both to the console and a log file (optional).

Two levels of information can be printed: `INFO` and `DEBUG`. `INFO` level displays a subset of the information printed by `DEBUG` level.

A logger can be created in your script with
```
from hrl.utils import setup_logger
logger = setup_logger(<level>, <log_file>)
```
And printouts can be added to the logger with:
```
logger.info(<message>) # printed out only in INFO level
logger.debug(<message>) # printed out in both INFO and DEBUG levels

