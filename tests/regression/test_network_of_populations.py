import json
import random
import tempfile

import numpy
import pandas as pd
import pytest

from simple_network_sim import common, network_of_populations as np, loaders
from tests.utils import create_baseline, calculateInfectiousOverTime


def _assert_baseline(result, force_update=False):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
        json.dump(result, fp, indent=4)
        baseline_filename = create_baseline(fp.name, force_update=force_update)

    with open(baseline_filename) as fp:
        assert result == pytest.approx(json.load(fp))


def _assert_baseline_dataframe(result, force_update=False):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
        result.to_csv(fp.name, index=False)
        fp.flush()
        baseline_filename = create_baseline(fp.name, force_update=force_update)
        pd.testing.assert_frame_equal(result, pd.read_csv(baseline_filename, dtype=np.RESULT_DTYPES))


def test_basic_simulation(data_api):
    network, _ = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/start-end-date", "start-end-date"),
    )

    df, issues = np.basicSimulationInternalAgeStructure(network, {"S08000016": {"[17,70)": 10.0}}, numpy.random.default_rng(123))
    result = calculateInfectiousOverTime(df, network.infectiousStates)

    _assert_baseline(result)
    assert not issues


def test_basic_simulation_with_dampening(data_api):
    network, _ = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/start-end-date", "start-end-date"),
        data_api.read_table("human/movement-multipliers", "movement-multipliers"),
    )

    df, issues = np.basicSimulationInternalAgeStructure(network, {"S08000016": {"[17,70)": 10.0}}, numpy.random.default_rng(123))
    result = calculateInfectiousOverTime(df, network.infectiousStates)

    _assert_baseline(result)
    assert not issues


def test_basic_simulation_stochastic(data_api_stochastic):
    network, _ = np.createNetworkOfPopulation(
        data_api_stochastic.read_table("human/compartment-transition", "compartment-transition"),
        data_api_stochastic.read_table("human/population", "population"),
        data_api_stochastic.read_table("human/commutes", "commutes"),
        data_api_stochastic.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api_stochastic.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api_stochastic.read_table("human/infection-probability", "infection-probability"),
        data_api_stochastic.read_table("human/initial-infections", "initial-infections"),
        data_api_stochastic.read_table("human/trials", "trials"),
        data_api_stochastic.read_table("human/start-end-date", "start-end-date"),
        data_api_stochastic.read_table("human/movement-multipliers", "movement-multipliers"),
        data_api_stochastic.read_table("human/stochastic-mode", "stochastic-mode"),
    )
    seed = loaders.readRandomSeed(data_api_stochastic.read_table("human/random-seed", "random-seed"))
    result, issues = np.basicSimulationInternalAgeStructure(network, {"S08000016": {"[17,70)": 10}}, numpy.random.default_rng(seed))

    _assert_baseline_dataframe(result)
    assert not issues


def test_basic_simulation_100_runs(data_api):
    network, _ = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/start-end-date", "start-end-date"),
    )

    runs = []
    rand = random.Random(1)
    issues = []
    for _ in range(100):
        regions = rand.choices(list(network.graph.nodes()), k=1)
        assert network.initialState[regions[0]][("[17,70)", "E")] == 0
        df, new_issues = np.basicSimulationInternalAgeStructure(network, {regions[0]: {"[17,70)": 10.0}}, numpy.random.default_rng(123))
        result = calculateInfectiousOverTime(df, network.infectiousStates)
        result.pop()  # TODO: due to historical reasons we have to ignore the last entry
        runs.append(result)
        issues.extend(new_issues)
    result = common.generateMeanPlot(runs)

    _assert_baseline(result)
    assert not issues
