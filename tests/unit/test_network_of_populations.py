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


def test_basicSimulationInternalAgeStructure_infect_more_than_susceptible():
    graph = nx.DiGraph()
    graph.add_node("region1")

    with pytest.raises(AssertionError):
        np.basicSimulationInternalAgeStructure(
            rand=random.Random(1),
            graph=graph,
            numInfected=11.0,
            timeHorizon=100,
            genericInfection=0.1,
            ageInfectionMatrix={"m": {"m": 0.2}},
            diseaseProgressionProbs={"S": {"S": 1.0}},
            dictOfStates={0: {"region1": {("m", "S"): 10.0}}},
        )


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


def test_internalStateDiseaseUpdate_does_not_affect_S_and_E():
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


@pytest.mark.parametrize("susceptible", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("infectious", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("asymptomatic", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("contact_rate", [0.0, 0.2, 1.0, 3.0])
def test_doInternalInfectionProcess_simple(susceptible, infectious, asymptomatic, contact_rate):
    current_state = {("m", "S"): susceptible, ("m", "A"): asymptomatic, ("m", "I"): infectious}
    age_matrix = {"m": {"m": contact_rate}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m"], 0)

    probability_of_susceptible = susceptible / (susceptible + infectious + asymptomatic)
    contacts = contact_rate * (asymptomatic + infectious)
    assert new_infected["m"] == probability_of_susceptible * contacts


def test_doInternalInfectionProcess_empty_age_group():
    current_state = {("m", "S"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    age_matrix = {"m": {"m": 0.0}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m"], 0)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_contact():
    current_state = {("m", "S"): 500.0, ("m", "A"): 100.0, ("m", "I"): 100.0}
    age_matrix = {"m": {"m": 0.0}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m"], 0)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_susceptibles():
    current_state = {("m", "S"): 0.0, ("m", "A"): 100.0, ("m", "I"): 100.0}
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m"], 0)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_infectious():
    current_state = {("m", "S"): 300.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m"], 0)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_only_A_and_I_count_as_infectious():
    current_state = {
        ("m", "S"): 300.0,
        ("m", "E"): 100.0,
        ("m", "A"): 0.0,
        ("m", "I"): 0.0,
        ("m", "H"): 100.0,
        ("m", "D"): 100.0,
        ("m", "R"): 100.0,
    }
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m"], 0)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_between_ages():
    current_state = {
        ("m", "S"): 20.0,
        ("m", "A"): 150.0,
        ("m", "I"): 300,
        ("o", "S"): 15.0,
        ("o", "A"): 200.0,
        ("o", "I"): 100.0,
    }
    age_matrix = {"m": {"m": 0.2, "o": 0.5}, "o": {"o": 0.3, "m": 0.5}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix, ["m", "o"], 0)

    assert new_infected["m"] == (20.0 / 470.0) * ((450.0 * 0.2) + (300.0 * 0.5))
    assert new_infected["o"] == (15.0 / 315.0) * ((300.0 * 0.3) + (450.0 * 0.5))


def test_doInteralInfectionProcessAllNodes_single_compartment():
    states = {
        0: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}},
        1: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}},
    }
    age_matrix = {"m": {"m": 0.2}}

    np.doInteralInfectionProcessAllNodes(states, age_matrix, ["m"], 0)

    new_infected = (300.0 / 400.0) * (0.2 * 100.0)  # 15.0
    assert states[1]["region1"] == {("m", "S"): 300.0 - new_infected, ("m", "E"): new_infected, ("m", "A"): 100.0, ("m", "I"): 0.0}


def test_doInteralInfectionProcessAllNodes_large_num_infected_raises_exception():
    states = {
        0: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}},
        1: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}},
    }
    age_matrix = {"m": {"m": 5.0}}

    with pytest.raises(AssertionError):
        np.doInteralInfectionProcessAllNodes(states, age_matrix, ["m"], 0)


def test_doIncomingInfectionsByNode_no_susceptibles():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2")

    state = {
        "r1": {("m", "S"): 0.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 0.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 5.0},
    }

    totalIncomingInfectionsByNode = np.doIncomingInfectionsByNode(graph, state)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 0.0}


def test_doIncomingInfectionsByNode_no_connections():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")

    state = {
        "r1": {("m", "S"): 100.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 100.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 5.0},
    }

    totalIncomingInfectionsByNode = np.doIncomingInfectionsByNode(graph, state)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 0.0}


def test_doIncomingInfectionsByNode_no_weight():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2")

    state = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }

    totalIncomingInfectionsByNode = np.doIncomingInfectionsByNode(graph, state)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 1.0 * 0.1 * 0.8}


def test_doIncomingInfectionsByNode_weight_given():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=0.5)

    state = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }

    totalIncomingInfectionsByNode = np.doIncomingInfectionsByNode(graph, state)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 0.5 * 0.1 * 0.8}


def test_doBetweenInfectionAgeStructured():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=0.5)

    states = {
        0: {
            "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
            "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
        },
        1: {
            "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
            "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
        },
    }
    original_states = copy.deepcopy(states)

    np.doBetweenInfectionAgeStructured(graph, states, 0, 0.1)

    new_infected = 0.5 * 0.1 * 0.8
    assert states[1]["r2"] == {("m", "S"): 80.0 - new_infected, ("m", "E"): new_infected, ("m", "A"): 10.0, ("m", "I"): 10.0}
    assert states[1]["r1"] == original_states[1]["r1"]


def test_doBetweenInfectionAgeStructured_caps_number_of_infections():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=60)

    states = {
        0: {
            "r1": {("m", "S"): 0.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0},
            "r2": {("m", "S"): 30.0, ("m", "E"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0},
        },
        1: {
            "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
            "r2": {("m", "S"): 30.0, ("m", "E"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0},
        },
    }
    original_states = copy.deepcopy(states)

    np.doBetweenInfectionAgeStructured(graph, states, 0, 0.1)

    assert states[1]["r2"] == {("m", "S"): 0.0, ("m", "E"): 30.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    assert states[1]["r1"] == original_states[1]["r1"]


def test_distributeInfections_cap_infections():
    state = {("m", "S"): 20.0}

    infections = np.distributeInfections(state, 100)

    assert infections == {"m": 20.0}


def test_distributeInfections_single_age_always_gets_full_infections():
    state = {("m", "S"): 20.0}

    infections = np.distributeInfections(state, 10)

    assert infections == {"m": 10.0}


def test_distributeInfections_infect_proportional_to_susceptibles_in_age_group():
    state = {("m", "S"): 20.0, ("o", "S"): 30.0, ("y", "S"): 40.0}

    infections = np.distributeInfections(state, 60)

    assert infections == {"m": (20.0 / 90.0) * 60, "o": (30.0 / 90.0) * 60, "y": (40.0 / 90.0) * 60}
