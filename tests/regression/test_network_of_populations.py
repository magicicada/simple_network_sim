import json
import random
import tempfile
import pandas as pd

import pytest

from simple_network_sim import common, network_of_populations as np
from tests.utils import create_baseline, calculateInfectiousOverTime


def _assert_baseline(result, force_update=False):
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        json.dump(result, fp, indent=4)
        baseline_filename = create_baseline(fp.name, force_update=force_update)

    with open(baseline_filename) as fp:
        assert result == pytest.approx(json.load(fp))


def _assert_baseline_dataframe(result, force_update=False):
    with tempfile.NamedTemporaryFile(mode="w") as fp:
        baseline_filename = create_baseline(fp.name, force_update=force_update)
        pd.testing.assert_frame_equal(result, pd.read_csv(baseline_filename, dtype=np.RESULT_DTYPES))


def test_basic_simulation(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/trials", "trials"),
    )

    result = calculateInfectiousOverTime(
        np.basicSimulationInternalAgeStructure(network, 200, {"S08000016": {"[17,70)": 10.0}}),
        network.infectiousStates
    )

    _assert_baseline(result)


def test_basic_simulation_with_dampening(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/movement-multipliers", "movement-multipliers"),
    )

    result = calculateInfectiousOverTime(
        np.basicSimulationInternalAgeStructure(network, 200, {"S08000016": {"[17,70)": 10.0}}),
        network.infectiousStates,
    )

    _assert_baseline(result)


def test_basic_simulation_stochastic(data_api_stochastic):
    network = np.createNetworkOfPopulation(
        data_api_stochastic.read_table("human/compartment-transition", "compartment-transition"),
        data_api_stochastic.read_table("human/population", "population"),
        data_api_stochastic.read_table("human/commutes", "commutes"),
        data_api_stochastic.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api_stochastic.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api_stochastic.read_table("human/infection-probability", "infection-probability"),
        data_api_stochastic.read_table("human/initial-infections", "initial-infections"),
        data_api_stochastic.read_table("human/trials", "trials"),
        data_api_stochastic.read_table("human/movement-multipliers", "movement-multipliers"),
        data_api_stochastic.read_table("human/stochastic-mode", "stochastic-mode"),
        data_api_stochastic.read_table("human/random-seed", "random-seed"),
    )

    result = np.basicSimulationInternalAgeStructure(network, 200, {"S08000016": {"[17,70)": 10}})

    _assert_baseline_dataframe(result)


def test_basic_simulation_100_runs(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/trials", "trials"),
    )

    runs = []
    rand = random.Random(1)
    for _ in range(100):
        regions = rand.choices(list(network.graph.nodes()), k=1)
        assert network.initialState[regions[0]][("[17,70)", "E")] == 0
        result = calculateInfectiousOverTime(
            np.basicSimulationInternalAgeStructure(network, 200, {regions[0]: {"[17,70)": 10.0}}),
            network.infectiousStates,
        )
        result.pop()  # TODO: due to historical reasons we have to ignore the last entry
        runs.append(result)
    result = common.generateMeanPlot(runs)

    _assert_baseline(result)
