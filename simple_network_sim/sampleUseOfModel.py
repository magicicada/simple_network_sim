"""
This is the main module used to run simulations based on network of populations
"""
import argparse
import logging
import logging.config
import sys
import time

from data_pipeline_api import standard_api
import pandas as pd
from pathlib import Path
from typing import Optional, List, NamedTuple
from functools import reduce

from . import common
from . import network_of_populations as ss

# Default logger, used if module not called as __main__
logger = logging.getLogger(__name__)


def main(argv):
    t0 = time.time()

    args = build_args(argv)
    setup_logger(args)
    logger.info("Parameters\n%s", "\n".join(f"\t{key}={value}" for key, value in args._get_kwargs()))

    issues: List[standard_api.Issue] = []

    info = common.get_repo_info()
    if not info.git_sha:
        desc = "Not running from a git repo, so no git_sha associated with the run"
        logger.warning(desc)
        issues.append(standard_api.Issue(severity=10, description=desc))
    elif info.is_dirty:
        desc = "Running out of a dirty git repo"
        logger.warning(desc)
        issues.append(standard_api.Issue(severity=10, description=desc))
    with standard_api.StandardAPI(args.data_pipeline_config, uri=info.uri, git_sha=info.git_sha) as store:
        network = ss.createNetworkOfPopulation(
            store.read_table("human/compartment-transition", "compartment-transition"),
            store.read_table("human/population", "population"),
            store.read_table("human/commutes", "commutes"),
            store.read_table("human/mixing-matrix", "mixing-matrix"),
            store.read_table("human/infectious-compartments", "infectious-compartments"),
            store.read_table("human/infection-probability", "infection-probability"),
            store.read_table("human/initial-infections", "initial-infections"),
            store.read_table("human/trials", "trials"),
            store.read_table("human/movement-multipliers", "movement-multipliers") if args.use_movement_multipliers else None,
            store.read_table("human/stochastic-mode", "stochastic-mode"),
            store.read_table("human/random-seed", "random-seed"),
        )

        results = runSimulation(network, args.time)
        aggregated = aggregateResults(results)

        logger.info("Writing output")
        store.write_table(
            "output/simple_network_sim/outbreak-timeseries",
            "outbreak-timeseries",
            aggregated.output,
            issues=issues,
            description=aggregated.description,
        )

        logger.info("Took %.2fs to run the simulation.", time.time() - t0)
        logger.info(
            "Use `python -m simple_network_sim.network_of_populations.visualisation -h` to find out how to take "
            "a peak what you just ran. You will need use the access-<hash>.yaml file that was created by this run."
        )


def runSimulation(
    network: ss.NetworkOfPopulation,
    max_time: int,
) -> List[pd.DataFrame]:
    """Run pre-created network

    :param network: object representing the network of populations
    :param max_time: Maximum time for simulation
    :return: Result runs for all trials of the simulation
    """
    results = []
    for i in range(network.trials):
        logger.info("Running simulation (%s/%s)", i + 1, network.trials)
        result = ss.basicSimulationInternalAgeStructure(network, max_time, network.initialInfections)
        results.append(result)

    return results


class AggregatedResults(NamedTuple):
    """
    This object contains the results of a simulation and a small description
    """
    output: pd.DataFrame
    description: str = "A dataframe of the number of people in each node, compartment and age over time"


def aggregateResults(results: List[pd.DataFrame]) -> AggregatedResults:
    """Aggregate results from runs

    :param results: result runs from runSimulation
    :return: Averaged number of infection through time, for all trial
    """
    if len(results) == 1:
        aggregated = results[0]
    else:
        results = [result.set_index(["time", "node", "age", "state"]).total for result in results]
        average = reduce(lambda x, y: x.add(y), results) / len(results)

        aggregated = average.reset_index()
    # The data pipeline API doesn't support categorical data, so we need to convert these to strings
    return AggregatedResults(output=aggregated.astype({"node": "str", "age": "str", "state": "str"}))


def setup_logger(args: Optional[argparse.Namespace] = None) -> None:
    """Configure package-level logger instance.
    
    :param args: argparse.Namespace
        args.logfile (pathlib.Path) is used to create a logfile if present
        args.quiet and args.debug control logging level to sys.stderr

    This function can be called without args, in which case it configures the
    package logger to write WARNING and above to STDERR.

    When called with args, it uses args.logfile to determine if logs (by
    default, INFO and above) should be written to a file, and the path of
    that file. args.quiet and args.debug are used to control reporting
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
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {__package__: {"handlers": ["stderr"], "level": "DEBUG"}},
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

    # Set STDERR/logfile levels if args.quiet/args.debug specified
    if args is not None and args.quiet:
        logconf["handlers"]["stderr"]["level"] = "WARNING"
    elif args is not None and args.debug:
        logconf["handlers"]["stderr"]["level"] = "DEBUG"
        if "logfile" in logconf["handlers"]:
            logconf["handlers"]["logfile"]["level"] = "DEBUG"

    # Configure logger
    logging.config.dictConfig(logconf)


def build_args(argv):
    """Return parsed CLI arguments as argparse.Namespace.

    :param argv: CLI arguments
    :type argv: list
    """

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
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Prints only warnings to stderr",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Provide debug output to STDERR"
    )
    parser.add_argument(
        "-c",
        "--data-pipeline-config",
        default="config.yaml",
        help="Base directory with the input parameters",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    # The logger name inherits from the package, if called as __main__
    logger = logging.getLogger(f"{__package__}.{__name__}")
    main(sys.argv[1:])
