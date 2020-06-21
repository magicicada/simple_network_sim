import json
import random
import tempfile

import pytest

from simple_network_sim import common, network_of_populations as np
from tests.utils import create_baseline, calculateInfectiousOverTime


def _assert_baseline(result, force_update=False):
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        json.dump(result, fp, indent=4)
        baseline_filename = create_baseline(fp.name, force_update=force_update)

    with open(baseline_filename) as fp:
        assert result == pytest.approx(json.load(fp))


def test_basic_simulation(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", version=1),
        data_api.read_table("human/population", version=1),
        data_api.read_table("human/commutes", version=1),
        data_api.read_table("human/mixing-matrix", version=1),
        data_api.read_table("human/infectious-compartments", version=1),
        data_api.read_table("human/infection-probability", version=1),
    )
    np.exposeRegions({"S08000016": {"[17,70)": 10.0}}, network.initialState)

    result = calculateInfectiousOverTime(np.basicSimulationInternalAgeStructure(network=network, timeHorizon=200), network.infectiousStates)

    _assert_baseline(result)


def test_basic_simulation_with_dampening(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", version=1),
        data_api.read_table("human/population", version=1),
        data_api.read_table("human/commutes", version=1),
        data_api.read_table("human/mixing-matrix", version=1),
        data_api.read_table("human/infectious-compartments", version=1),
        data_api.read_table("human/infection-probability", version=1),
        data_api.read_table("human/movement-multipliers", version=1),
    )
    np.exposeRegions({"S08000016": {"[17,70)": 10.0}}, network.initialState)

    result = calculateInfectiousOverTime(
        np.basicSimulationInternalAgeStructure(network=network, timeHorizon=200),
        network.infectiousStates,
    )

    _assert_baseline(result)


def test_basic_simulation_100_runs(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", version=1),
        data_api.read_table("human/population", version=1),
        data_api.read_table("human/commutes", version=1),
        data_api.read_table("human/mixing-matrix", version=1),
        data_api.read_table("human/infectious-compartments", version=1),
        data_api.read_table("human/infection-probability", version=1),
    )

    runs = []
    rand = random.Random(1)
    for _ in range(100):
        regions = rand.choices(list(network.graph.nodes()), k=1)
        # This was added for backwards compatibility. Notice that ("m", "S") diminishes at each run.
        # TODO: make sure the states[0] is always reset after each run or that a new state is created before running
        #       exposeRegions
        network.initialState[regions[0]][("[17,70)", "E")] = 0
        np.exposeRegions({regions[0]: {"[17,70)": 10.0}}, network.initialState)
        result = calculateInfectiousOverTime(
            np.basicSimulationInternalAgeStructure(network=network, timeHorizon=200),
            network.infectiousStates,
        )
        result.pop()  # TODO: due to historical reasons we have to ignore the last entry
        runs.append(result)
    result = common.generateMeanPlot(runs)

    _assert_baseline(result)
