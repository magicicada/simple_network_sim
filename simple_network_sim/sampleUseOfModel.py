import argparse
import copy
import logging
import logging.config
import sys
import time

from pathlib import Path
from typing import Optional

from data_pipeline_api.api import API, DataAccess, ParameterRead
from data_pipeline_api.file_system_data_access import FileSystemDataAccess

from . import network_of_populations as ss, loaders

# Default logger, used if module not called as __main__
logger = logging.getLogger(__name__)


def main(argv):
    t0 = time.time()

    args = build_args(argv)
    setup_logger(args)
    logger.info("Parameters\n%s", "\n".join(f"\t{key}={value}" for key, value in args._get_kwargs()))

    api = API(FileSystemDataAccess(args.base_data_dir, args.metadata_file))

    network = ss.createNetworkOfPopulation(
        api.read_table("human/compartment-transition", version=1),
        api.read_table("human/population", version=1),
        api.read_table("human/commutes", version=1),
        api.read_table("human/mixing-matrix", version=1),
        api.read_table("human/infectious-compartments", version=1),
        api.read_table("human/movement-multipliers", version=1) if args.use_movement_multipliers else None,
    )

    initialInfections = []
    if args.cmd == "seeded":
        with open(args.input) as fp:
            initialInfections.append(loaders.readInitialInfections(fp))
    elif args.cmd == "random":
        for _ in range(args.trials):
            initialInfections.append(ss.randomlyInfectRegions(network, args.regions, args.age_groups, args.infected))

    results = runSimulation(network, args.time, args.trials, initialInfections)
    plot_states = args.plot_states.split(",") if args.plot_states else None
    plot_nodes = args.plot_nodes.split(",") if args.plot_nodes else None
    # TODO: replace the current .csv file that's saved as part of the results with api.write table. This was not done
    #       yet while we wait for the next data API version
    saveResults(results, args.output_prefix, plot_states, plot_nodes)
    api.write_table(results, "output/simple_network_sim/outbreak-timeseries", version=1)

    logger.info("Took %.2fs to run the simulation.", time.time() - t0)
    api.close()


def runSimulation(network, max_time, trials, initialInfections):
    """Run pre-created network

    :param network: object representing the network of populations
    :type network: A NetworkOfPopulation object
    :param max_time: Maximum time for simulation
    :type max_time: int
    :param trials: Number of simulation trials
    :type trials: int
    :param initialInfections: List of initial infection. If seeded, only one
    :type initialInfections: list
    :return: Averaged number of infection through time, through trials
    :rtype: list
    """
    aggregated = None

    for i in range(trials):
        disposableNetwork = copy.deepcopy(network)

        ss.exposeRegions(initialInfections[i], disposableNetwork.states[0])
        ss.basicSimulationInternalAgeStructure(disposableNetwork, max_time)
        indexed = ss.modelStatesToPandas(disposableNetwork.states).set_index(["time", "node", "age", "state"])

        if aggregated is None:
            aggregated = indexed
        else:
            aggregated.total += indexed.total

    averaged = aggregated.reset_index()
    averaged.total /= trials

    return averaged


def saveResults(results, output_prefix, plot_states, plot_nodes):
    """Save result from simulation to csv (raw results) and pdf (plot per node).

    :param results: Results from simulation
    :type results: pd.DataFrame
    :param output_prefix: Prefix for output file
    :type output_prefix: str
    :param plot_states: plots one curve per state listed (None means all states)
    :type plot_states: list (of disease states).
    :param plot_nodes: creates one plot per node listed (None means all nodes)
    :type plot_nodes: list (of region names).
    """
    filename = f"{output_prefix}-{int(time.time())}"

    results.to_csv(f"{filename}.csv", index=False)
    ss.plotStates(results, states=plot_states, nodes=plot_nodes).savefig(f"{filename}.pdf", dpi=300)

    logger.info("Read the dataframe from: %s.csv", filename)
    logger.info("Open the visualisation from: %s.pdf", filename)


def setup_logger(args: Optional[argparse.Namespace] = None) -> None:
    """Configure package-level logger instance.
    
    :param args: argparse.Namespace
        args.logfile (pathlib.Path) is used to create a logfile if present
        args.verbose and args.debug control logging level to sys.stderr

    This function can be called without args, in which case it configures the
    package logger to write WARNING and above to STDERR.

    When called with args, it uses args.logfile to determine if logs (by
    default, INFO and above) should be written to a file, and the path of
    that file. args.verbose and args.debug are used to control reporting
    level.
    """
    # Dictionary to define logging configuration
    logconf = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            },
        },
        "handlers": {
            "stderr": {
                "level": "WARNING",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {__package__: {"handlers": ["stderr"], "level": "DEBUG",},},
    }

    # If args.logpath is specified, add logfile
    if args is not None and args.logfile is not None:
        logdir = args.logfile.parents[0]
        # If the logfile is going in another directory, we must
        # create/check if the directory is there
        try:
            if not logdir == Path.cwd():
                logdir.mkdir(exist_ok=True)
        except OSError:
            logger.error("Could not create %s for logging", logdir, exc_info=True)
            raise SystemExit(1)  # Substitute meaningful error code when known
        # Add logfile configuration
        logconf["handlers"]["logfile"] = {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": str(args.logfile),
            "encoding": "utf8",
        }
        logconf["loggers"][__package__]["handlers"].append("logfile")

    # Set STDERR/logfile levels if args.verbose/args.debug specified
    if args is not None and args.verbose:
        logconf["handlers"]["stderr"]["level"] = "INFO"
    elif args is not None and args.debug:
        logconf["handlers"]["stderr"]["level"] = "DEBUG"
        logconf["handlers"]["logfile"]["level"] = "DEBUG"

    # Configure logger
    logging.config.dictConfig(logconf)


def build_args(argv, inputFilesFolder="sample_input_files"):
    """Return parsed CLI arguments as argparse.Namespace.

    :param argv: CLI arguments
    :type argv: list
    :param inputFilesFolder: Folder name with input files
    :type inputFilesFolder: str
    """
    sampledir = Path(__file__).parents[1] / inputFilesFolder

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Uses the deterministic network of populations model to simulation the disease progression",
    )
    parser.add_argument(
        "--use-movement-multipliers",
        action="store_true",
        help="By enabling this parameter you can adjust dampening or heightening people movement through time",
    )
    parser.add_argument(
        "--time",
        default=200,
        type=int,
        help="The number of time steps to take for each simulation",
    )
    parser.add_argument(
        "-l",
        "--logfile",
        dest="logfile",
        default=None,
        type=Path,
        help="Path for logging output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Provide verbose output to STDERR",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Provide debug output to STDERR"
    )
    parser.add_argument(
        "--plot-nodes",
        default=None,
        metavar="nodes,[nodes,...]",
        help="Comma-separated list of nodes to plot. All nodes will be plotted if not provided."
    )
    parser.add_argument(
        "--plot-states",
        default=None,
        metavar="states,[states,...]",
        help="Comma-separated list of states to plot. All states will be plotted if not provided."
    )
    parser.add_argument(
        "-b",
        "--base-data-dir",
        default="data_pipeline_inputs",
        help="Base directory with the input paramters",
    )
    parser.add_argument(
        "-m",
        "--metadata-file",
        default="metadata.toml",
        help="Data API interaction log"
    )

    sp = parser.add_subparsers(dest="cmd", required=True)

    # Parameters when using the random infection approach
    randomCmd = sp.add_parser(
        "random",
        help="Randomly pick regions to infect",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    randomCmd.add_argument("--regions", default=1, help="Number of regions to infect")
    randomCmd.add_argument(
        "--age-groups", nargs="+", default=["[17,70)"], help="Age groups to infect"
    )
    randomCmd.add_argument(
        "--trials", default=100, type=int, help="Number of experiments to run"
    )
    randomCmd.add_argument(
        "--infected",
        default=100,
        type=int,
        help="Number of infected people in each region/age group",
    )
    randomCmd.add_argument(
        "output_prefix",
        type=str,
        help="Prefix used when exporting the dataframe and plot",
    )

    # Parameters when using the seeded infection approach
    seededCmd = sp.add_parser(
        "seeded",
        help="Use a seed file with infected regions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    seededCmd.add_argument(
        "--trials", default=1, type=int, help="Number of experiments to run"
    )
    seededCmd.add_argument(
        "input", type=Path, help="File name with the seed region seeds"
    )
    seededCmd.add_argument(
        "output_prefix",
        type=str,
        help="Prefix use when exporting the dataframe and plot",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    # The logger name inherits from the package, if called as __main__
    logger = logging.getLogger(f"{__package__}.{__name__}")
    main(sys.argv[1:])
