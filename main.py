import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-replay", action="store_true")
    parser.add_argument("--hindsight", action="store_true")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    pass

if __name__ == "__main__":
    main()
