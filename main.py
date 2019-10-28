import argparse


def main(args):
    parser.add_argument("-shard_size", default=2000, type=int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)