"""
Visualisation tool for network of populations
"""
# pylint: disable=import-error
import argparse
import logging
import math
import sys
from pathlib import Path

from data_pipeline_api.file_formats import object_file
import pandas as pd
import yaml

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def plot_nodes(df, nodes=None, states=None, ncol=3, sharey=False, figsize=None, cmap=None):
    """
    Plots a grid of plots, one plot per node, filtered by disease progression states (each states will be a line). The
    graphs are all Number of People x Time

    :param df: pandas DataFrame with node, time, state and total columns
    :param nodes: creates one plot per nodes listed (None means all nodes)
    :param df: pandas DataFrame with nodes, time, state and total columns
    :type df: pandas DataFrame
    :param nodes: creates one plot per node listed (None means all nodes)
    :type nodes: list (of region names).
    :param states: plots one curve per state listed (None means all states)
    :type states: list (of disease states).
    :param ncol: number of columns (the number of rows will be calculated to fit all graphs)
    :type ncol: int
    :param sharey: set to true if all plots should have the same y-axis
    :type sharey: bool
    :param figsize: select the size of each individual plot
    :type figsize:
    :param cmap: color map to use
    :type cmap:
    :return: returns a matplotlib figure
    :rtype: matplotlib figure
    """
    if nodes is None:
        nodes = df.node.unique().tolist()
    if states is None:
        states = df.state.unique().tolist()
    if cmap is None:
        cmap = ListedColormap(["#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#E69F00"])
    nrow = math.ceil(len(nodes) / ncol)
    if figsize is None:
        figsize = (20, nrow * 5)

    if not nodes:
        raise ValueError("nodes cannot be an empty list")
    if not states:
        raise ValueError("states cannot be an empty list")

    # pre filter by states
    df = df[df.state.isin(states)]

    fig, axes = plt.subplots(nrow, ncol, squeeze=False, constrained_layout=True, sharey=sharey, figsize=figsize)

    count = 0
    ax = None
    for i in range(nrow):
        for j in range(ncol):
            if count < len(nodes):
                node = nodes[count]
                count += 1
                grouped = df[df.node == node].groupby(["time", "state"]).sum()
                indexed = grouped.reset_index().pivot(index="time", columns="state", values="total")

                ax = axes[i, j]
                indexed.plot(ax=ax, legend=False, title=node, cmap=cmap)
                ax.set_ylabel("Number of People")
                ax.set_xlabel("Time")

    assert ax is not None, "ax was never assigned"
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    return fig


def read_output(data_product: str, path: str) -> pd.DataFrame:
    """
    Read a data product from the run in path

    :param data_product: the name of the data_product to read
    :param path: the path to read it from
    :return: The output data, loaded as a pandas DataFrame
    """
    with open(path) as fp:
        access_log = yaml.safe_load(fp)
    outputs = list(
        filter(
            lambda x: x["type"] == "write" and x["call_metadata"]["data_product"] == data_product,
            access_log["io"]
        )
    )
    assert len(outputs) == 1, f"More than one output selected: {outputs}"

    output_path = Path(access_log["data_directory"]) / Path(outputs[0]["access_metadata"]["filename"])
    with open(output_path, "rb") as fp:
        return object_file.read_table(output_path, "outbreak-timeseries")


def build_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads the access*.yaml files and outputs graphs for a given run",
    )

    parser.add_argument(
        "--nodes",
        default=None,
        metavar="nodes,[nodes,...]",
        help="Comma-separated list of nodes to plot. All nodes will be plotted if not provided."
    )
    parser.add_argument(
        "--states",
        default=None,
        metavar="states,[states,...]",
        help="Comma-separated list of states to plot. All states will be plotted if not provided."
    )
    parser.add_argument(
        "--share-y",
        default=False,
        action="store_true",
        help="Toggle this flag if you want all y-axis to be shared",
    )
    parser.add_argument(
        "--data-product",
        default="output/simple_network_sim/outbreak-timeseries",
        help="Use this to select which output file to read, in case more than one is available"
    )

    parser.add_argument("access_log_path", type=str, help="Path to a access log file")

    return parser.parse_args(argv)


def main(argv):
    """
    This is the main function of the visualisation tool. The tool outputs graphs for a given network of populations run
    """
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )
    args = build_args(argv)
    df = read_output(args.data_product, args.access_log_path)
    plot_nodes(
        df,
        args.nodes.split(",") if args.nodes else None,
        args.states.split(",") if args.states else None,
        sharey=args.share_y,
    )
    plt.show()


if __name__ == "__main__":
    # The logger name inherits from the package, if called as __main__
    logger = logging.getLogger(f"{__package__}.{__name__}")
    main(sys.argv[1:])
