import argparse
from data.generator import DataGenerator
from constants import TRAIN_SIZE, VAL_SIZE, TEST_SIZE


def main(args):
    generator = DataGenerator(args.data_dir)
    generator.generate_split("train", TRAIN_SIZE)
    generator.generate_split("val", VAL_SIZE)
    generator.generate_split("test", TEST_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/synthetic",
        help="Directory to save generated data",
    )

    args = parser.parse_args()
    main(args)
