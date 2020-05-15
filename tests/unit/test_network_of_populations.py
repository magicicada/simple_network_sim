import copy
import itertools
import random

import networkx as nx
import pytest

from simple_network_sim import network_of_populations as np, loaders


def _count_people_per_region(state):
    return [sum(region.values()) for region in state.values()]


@pytest.mark.parametrize("seed", [2, 3])
@pytest.mark.parametrize("num_infected", [0, 10])
@pytest.mark.parametrize("generic_infection", [0.1, 1.0, 1.5])
def test_basicSimulationInternalAgeStructure_invariants(
    age_transitions,
    demographics,
    commute_moves,
    compartment_names,
    age_infection_matrix,
    num_infected,
    generic_infection,
    seed,
):
    age_to_trans = np.setUpParametersAges(loaders.readParametersAgeStructured(age_transitions))
    population = loaders.readPopulationAgeStructured(demographics)
    graph = loaders.genGraphFromContactFile(commute_moves)
    states = np.setupInternalPopulations(graph, compartment_names, list(age_to_trans.keys()), population)
    old_graph = copy.deepcopy(graph)
    old_age_to_trans = copy.deepcopy(age_to_trans)
    initial_population = sum(_count_people_per_region(states[0])) + num_infected

    np.basicSimulationInternalAgeStructure(
        rand=random.Random(seed),
        graph=graph,
        numInfected=num_infected,
        timeHorizon=50,
        genericInfection=generic_infection,
        ageInfectionMatrix=age_infection_matrix,
        diseaseProgressionProbs=age_to_trans,
        dictOfStates=states,
    )

    # population remains constant
    assert all([sum(_count_people_per_region(state)) == pytest.approx(initial_population) for state in states.values()])

    # the graph is unchanged
    assert nx.is_isomorphic(old_graph, graph)

    # infection matrix is unchanged
    assert age_to_trans == old_age_to_trans


def test_internalStateDiseaseUpdate_one_transition():
    current_state = {("o", "E"): 100.0, ("o", "A"): 0.0}
    probs = {"o": {"E": {"A": 0.4, "E": 0.6}, "A": {"A": 1.0}}}

    new_state = np.internalStateDiseaseUpdate(current_state, probs)

    assert new_state == {("o", "E"): 60.0, ("o", "A"): 40.0}


def test_internalStateDiseaseUpdate_no_transitions():
    current_state = {("o", "E"): 100.0, ("o", "A"): 0.0}
    probs = {"o": {"E": {"E": 1.0}, "A": {"A": 1.0}}}

    new_state = np.internalStateDiseaseUpdate(current_state, probs)

    assert new_state == {("o", "E"): 100.0, ("o", "A"): 0.0}


def test_internalStateDiseaseUpdate_progression_only():
    current_state = {("o", "S"): 100.0, ("o", "E"): 0.0}
    probs = {"o": {"S": {"E": 0.4, "S": 0.6}, "E": {"E": 1.0}}}

    new_state = np.internalStateDiseaseUpdate(current_state, probs)

    assert new_state == {("o", "S"): 100.0, ("o", "E"): 0.0}


def test_doInternalProgressionAllNodes_create_new_state():
    states = {0: {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}}}
    probs = {"o": {"E": {"A": 0.4, "E": 0.6}, "A": {"A": 1.0}}}

    np.doInternalProgressionAllNodes(states, 0, probs)

    expected = {
        0: {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}},
        1: {"region1": {("o", "E"): 60.0, ("o", "A"): 40.0}},
    }

    assert states == expected


def test_doInternalProgressionAllNodes_overwrite_state():
    states = {
        0: {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}},
        1: {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}},
    }
    probs = {"o": {"E": {"A": 0.4, "E": 0.6}, "A": {"A": 1.0}}}

    np.doInternalProgressionAllNodes(states, 0, probs)

    expected = {
        0: {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}},
        1: {"region1": {("o", "E"): 60.0, ("o", "A"): 40.0}},
    }

    assert states == expected
