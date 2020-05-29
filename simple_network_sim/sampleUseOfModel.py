import argparse
import copy
import logging
import os
import random
import sys
import time

import matplotlib.pyplot as plt

from . import common, network_of_populations as ss, loaders

logger = logging.getLogger(__name__)


def main(argv):
    t0 = time.time()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")

    args = build_args(argv)
    logger.info("Parameters\n%s", "\n".join(f"\t{key}={value}" for key, value in args._get_kwargs()))

    network = ss.createNetworkOfPopulation(
        disasesProgressionFn=args.compartment_transition,
        populationFn=args.population,
        graphFn=args.commutes,
        ageInfectionMatrixFn=args.mixing_matrix,
        movementMultipliersFn=args.movement_multipliers,
    )

    basicPlots = []
    infections = {}
    if args.cmd == "seeded":
        with open(args.input) as fp:
            infections = loaders.readInitialInfections(fp)
    for _ in range(args.trials):
        disposableNetwork = copy.deepcopy(network)

        if args.cmd == "random":
            # If random, pick new infections at each iteration
            infections = {}
            for regionID in random.choices(list(disposableNetwork.graph.nodes()), k=args.regions):
                infections[regionID] = {}
                for age in args.age_groups:
                    infections[regionID][age] = args.infected

        ss.exposeRegions(infections, disposableNetwork.states[0])
        basicPlots.append(ss.basicSimulationInternalAgeStructure(disposableNetwork, args.time))

    plt.plot(common.generateMeanPlot(basicPlots), color ='dodgerblue', label='basic')

    plt.savefig(args.output)
    logger.info("Took %.2fs to run the simulation", time.time() - t0)


def build_args(argv):
    sampledir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "sample_input_files"))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Uses the deterministic network of populations model to simulation the disease progression",
    )
    parser.add_argument(
        "--compartment-transition",
        default=os.path.join(sampledir, "compartmentTransitionByAge.csv"),
        help="Epidemiological rate parameters for movement within the compartmental model.",
    )
    parser.add_argument(
        "--population",
        default=os.path.join(sampledir, "sample_hb2019_pop_est_2018_row_based.csv"),
        help="This file contains age-and-sex-stratified population numbers by geographic unit.",
    )
    parser.add_argument(
        "--commutes",
        default=os.path.join(sampledir, "sample_scotHB_commute_moves_wu01.csv"),
        help="This contains origin-destination flow data during peacetime for health boards",
    )
    parser.add_argument(
        "--mixing-matrix",
        default=os.path.join(sampledir, "simplified_age_infection_matrix.csv"),
        help="This is a sample square matrix of mixing - each column and row header is an age category.",
    )
    parser.add_argument(
        "--movement-multipliers",
        help="By using this parameter you can adjust dampening or heightening people movement through time",
    )
    parser.add_argument("--time", default=200, type=int, help="The number of time steps to take for each simulation")

    sp = parser.add_subparsers(dest="cmd", required=True)

    # Parameters when using the random infection approach
    randomCmd = sp.add_parser("random", help="Randomly pick regions to infect", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    randomCmd.add_argument("--regions", default=1, help="Number of regions to infect")
    randomCmd.add_argument("--age-groups", nargs="+", default=["[17,70)"], help="Age groups to infect")
    randomCmd.add_argument("--trials", default=100, type=int, help="Number of experiments to run")
    randomCmd.add_argument("--infected", default=100, type=int, help="Number of infected people in each region/age group")
    randomCmd.add_argument("output", help="Name of the PDF file that will be created with the visualisation")

    # Parameters when using the seeded infection approach
    seededCmd = sp.add_parser("seeded", help="Use a seed file with infected regions", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    seededCmd.add_argument("--trials", default=1, type=int, help="Number of experiments to run")
    seededCmd.add_argument("input", help="File name with the seed region seeds")
    seededCmd.add_argument("output", help="Name of the PDF file that will be created with the visualisation")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
