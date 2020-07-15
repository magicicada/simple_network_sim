import argparse
import logging
import sys

from data_pipeline_api.file_formats import object_file
import h5py

logger = logging.getLogger(f"{__package__}.{__name__}")


def main(argv):
    parser = argparse.ArgumentParser(description="Converts an hdf5 tabular file used by the model into a csv file")
    parser.add_argument(
        "input_file",
        help="This is the path for the hdf5 input file",
    )
    parser.add_argument(
        "output_file",
        help="This is the path for the csv output file",
    )
    parser.add_argument(
        "-c",
        "--component",
        default=None,
        help="Component to use when reading the hdf5 file. By default, the first (alphabetically) component in the file will be used"
    )
    args = parser.parse_args(argv)

    if args.component is None:
        h5 = h5py.File(args.input_file, "r")
        component = sorted(h5.keys())[0]
        logger.info("Using default component: %s", component)
    else:
        component = args.component
    with open(args.input_file, "rb") as fp:
        df = object_file.read_table(fp, component)
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )
    main(sys.argv[1:])
