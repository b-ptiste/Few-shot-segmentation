import sys
import argparse
from typing import Optional
import logging
from configuration import CFG_EXPERIMENTS as cfg
from pathlib import Path

WANDB_AVAILABLE = False
try:
    #TODO: set to True the following line when we setup wandb for this project (create wandb project and connect it to the repo)
    WANDB_AVAILABLE = False
    import wandb
except ImportError:
    logging.warning("Could not import wandb. Disabling wandb.")
    pass
from code import run_experiment

def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-o", "--output-dir", type=str, default=Path(__file__).parent/"__output", help="Output directory")
    parser.add_argument("-nowb", "--no-wandb", action="store_true", help="Disable weights and biases")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    exp_id = 0
    assert exp_id in list(cfg.keys()), f"Experiment {exp_id} not found in configuration.py"
    print("running experiment {}".format(exp_id))
    print(cfg[exp_id])
    run_experiment(cfg[exp_id], cpu=args.cpu, no_wandb=args.no_wandb)