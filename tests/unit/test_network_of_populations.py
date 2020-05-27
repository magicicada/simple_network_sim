import copy
import itertools
import random
import tempfile

import networkx as nx
import pytest

from simple_network_sim import network_of_populations as np, loaders


def _count_people_per_region(state):
    return [sum(region.values()) for region in state.values()]

@pytest.mark.parametrize("region", ["S08000024", "S08000030"])
@pytest.mark.parametrize("seed", [2, 3])
@pytest.mark.parametrize("num_infected", [0, 10])
def test_basicSimulationInternalAgeStructure_invariants(demographicsFilename, commute_moves, compartmentTransitionsByAgeFilename, simplified_mixing_matrix, region, seed, num_infected):
    network = np.createNetworkOfPopulation(compartmentTransitionsByAgeFilename, demographicsFilename, commute_moves, simplified_mixing_matrix)
    np.exposeRegions({region: {"[0,17)": num_infected}}, network.states[0])

    initial_population = sum(_count_people_per_region(network.states[0]))
    old_network = copy.deepcopy(network)

    np.basicSimulationInternalAgeStructure(network=network, timeHorizon=50)

    # population remains constant
    assert all([sum(_count_people_per_region(state)) == pytest.approx(initial_population) for state in network.states.values()])

    # the graph is unchanged
    assert nx.is_isomorphic(old_network.graph, network.graph)

    # infection matrix is unchanged
    assert list(network.infectionMatrix) == list(old_network.infectionMatrix)
    for a in network.infectionMatrix:
        assert list(network.infectionMatrix[a]) == list(old_network.infectionMatrix[a])
        for b in network.infectionMatrix[a]:
            assert network.infectionMatrix[a][b] == old_network.infectionMatrix[a][b]


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


def test_doInternalProgressionAllNodes_e_to_a_progession():
    states = {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}}
    probs = {"o": {"E": {"A": 0.4, "E": 0.6}, "A": {"A": 1.0}}}

    progression = np.getInternalProgressionAllNodes(states, probs)

    assert progression == {"region1": {("o", "E"): 60.0, ("o", "A"): 40.0}}
    assert states == {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}}  # unchanged


@pytest.mark.parametrize("susceptible", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("infectious", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("asymptomatic", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("contact_rate", [0.0, 0.2, 1.0, 3.0])
def test_doInternalInfectionProcess_simple(susceptible, infectious, asymptomatic, contact_rate):
    current_state = {("m", "S"): susceptible, ("m", "A"): asymptomatic, ("m", "I"): infectious}
    age_matrix = {"m": {"m": contact_rate}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

    probability_of_susceptible = susceptible / (susceptible + infectious + asymptomatic)
    contacts = contact_rate * (asymptomatic + infectious)
    assert new_infected["m"] == probability_of_susceptible * contacts


def test_doInternalInfectionProcess_empty_age_group():
    current_state = {("m", "S"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    age_matrix = {"m": {"m": 0.0}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_contact():
    current_state = {("m", "S"): 500.0, ("m", "A"): 100.0, ("m", "I"): 100.0}
    age_matrix = {"m": {"m": 0.0}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_susceptibles():
    current_state = {("m", "S"): 0.0, ("m", "A"): 100.0, ("m", "I"): 100.0}
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_infectious():
    current_state = {("m", "S"): 300.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

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

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

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

    new_infected = np.doInternalInfectionProcess(current_state, age_matrix)

    assert new_infected["m"] == (20.0 / 470.0) * ((450.0 * 0.2) + (300.0 * 0.5))
    assert new_infected["o"] == (15.0 / 315.0) * ((300.0 * 0.3) + (450.0 * 0.5))


def test_doInteralInfectionProcessAllNodes_single_compartment():
    states = {0: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}}}
    age_matrix = {"m": {"m": 0.2}}

    infections = np.getInternalInfection(states, age_matrix, 0)

    assert infections == {"region1": {"m": (300.0 / 400.0) * (0.2 * 100.0)}}
    assert states == {0: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}}}  # unchanged


def test_doInteralInfectionProcessAllNodes_large_num_infected_ignored():
    states = {0: {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}}}
    age_matrix = {"m": {"m": 5.0}}

    new_infected = np.getInternalInfection(states, age_matrix, 0)

    assert new_infected == {"region1": {"m": (300.0 / 400.0) * (100.0 * 5.0)}}


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
        }
    }
    original_states = copy.deepcopy(states)

    num_infections = np.getExternalInfections(graph, states, 0)

    assert num_infections == {"r1": {"m": 0.0}, "r2": {"m": 0.5 * 0.1 * 0.8}}
    assert states == original_states


def test_doBetweenInfectionAgeStructured_caps_number_of_infections():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=60)

    states = {
        0: {
            "r1": {("m", "S"): 0.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0},
            "r2": {("m", "S"): 30.0, ("m", "E"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0},
        }
    }
    original_states = copy.deepcopy(states)

    new_infections = np.getExternalInfections(graph, states, 0)

    assert new_infections == {"r1": {"m": 0.0}, "r2": {"m": 30.0}}
    assert states == original_states


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


def test_expose_infect_more_than_susceptible():
    region = {("m", "S"): 5.0, ("m", "E"): 0.0}

    with pytest.raises(AssertionError):
        np.expose("m", 10.0, region)


def test_expose():
    region = {("m", "S"): 15.0, ("m", "E"): 2.0}

    np.expose("m", 10.0, region)

    assert region == {("m", "S"): 5.0, ("m", "E"): 12.0}


def test_expose_change_only_desired_age():
    region = {("m", "S"): 15.0, ("m", "E"): 2.0, ("o", "S"): 10.0, ("o", "E"): 0.0}

    np.expose("m", 10.0, region)

    assert region == {("m", "S"): 5.0, ("m", "E"): 12.0, ("o", "S"): 10.0, ("o", "E"): 0.0}


def test_exposeRegion_distributes_multiple_ages():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0, ("o", "S"): 10.0, ("o", "E"): 0.0}}

    np.exposeRegions({"region1": {"m": 5.0, "o": 5.0}}, state)

    assert state == {"region1": {("m", "S"): 10.0, ("m", "E"): 5.0, ("o", "S"): 5.0, ("o", "E"): 5.0}}


def test_exposeRegion_requires_probabilities_fails_if_age_group_does_not_exist():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0}}

    with pytest.raises(KeyError):
        np.exposeRegions({"region1": {"m": 10.0, "o": 0.0}}, state)


def test_exposeRegion_multiple_regions():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0}, "region2": {("m", "S"): 15.0, ("m", "E"): 0.0}}

    np.exposeRegions({"region1": {"m": 10.0}, "region2": {"m": 10.0}}, state)

    assert state == {"region1": {("m", "S"): 5.0, ("m", "E"): 10.0}, "region2": {("m", "S"): 5.0, ("m", "E"): 10.0}}


def test_exposeRegion_only_desired_region():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0}, "region2": {("m", "S"): 15.0, ("m", "E"): 0.0}}

    np.exposeRegions({"region1": {"m": 10.0}}, state)

    assert state == {"region1": {("m", "S"): 5.0, ("m", "E"): 10.0}, "region2": {("m", "S"): 15.0, ("m", "E"): 0.0}}


def test_createNetworkOfPopulation(demographicsFilename, commute_moves, compartmentTransitionsByAgeFilename, simplified_mixing_matrix):
    network = np.createNetworkOfPopulation(compartmentTransitionsByAgeFilename, demographicsFilename, commute_moves, simplified_mixing_matrix)

    assert network.graph
    assert network.infectionMatrix
    assert network.states
    assert network.progression


def test_createNetworkOfPopulation_age_mismatch_matrix():
    with \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as progression, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as population, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as commutes, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as infectionMatrix:
        progression.write("age,src,dst,rate\n70+,E,E,1.0")
        progression.flush()
        population.write('Health_Board,Sex,Age,Total\nS08000015,Female,70+,31950')
        population.flush()
        commutes.write("S08000015,S08000015,100777")
        commutes.flush()
        infectionMatrix.write(",71+\n71+,1.0")
        infectionMatrix.flush()

        with pytest.raises(AssertionError):
            np.createNetworkOfPopulation(progression.name, population.name, commutes.name, infectionMatrix.name)


def test_createNetworkOfPopulation_age_mismatch_matrix_internal():
    with \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as progression, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as population, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as commutes, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as infectionMatrix:
        progression.write("age,src,dst,rate\n70+,E,E,1.0")
        progression.flush()
        population.write('Health_Board,Sex,Age,Total\nS08000015,Female,70+,31950')
        population.flush()
        commutes.write("S08000015,S08000015,100777")
        commutes.flush()
        infectionMatrix.write(",71+\n70+,1.0")
        infectionMatrix.flush()

        with pytest.raises(AssertionError):
            np.createNetworkOfPopulation(progression.name, population.name, commutes.name, infectionMatrix.name)


def test_createNetworkOfPopulation_age_mismatch_population():
    with \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as progression, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as population, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as commutes, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as infectionMatrix:
        progression.write("age,src,dst,rate\n70+,E,E,1.0")
        progression.flush()
        population.write('Health_Board,Sex,Age,Total\nS08000015,Female,71+,31950')
        population.flush()
        commutes.write("S08000015,S08000015,100777")
        commutes.flush()
        infectionMatrix.write(",70+\n70+,1.0")
        infectionMatrix.flush()

        with pytest.raises(AssertionError):
            np.createNetworkOfPopulation(progression.name, population.name, commutes.name, infectionMatrix.name)


def test_createNetworkOfPopulation_age_mismatch_progression():
    with \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as progression, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as population, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as commutes, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as infectionMatrix:
        progression.write("age,src,dst,rate\n71+,E,E,1.0")
        progression.flush()
        population.write('Health_Board,Sex,Age,Total\nS08000015,Female,70+,31950')
        population.flush()
        commutes.write("S08000015,S08000015,100777")
        commutes.flush()
        infectionMatrix.write(",70+\n70+,1.0")
        infectionMatrix.flush()

        with pytest.raises(AssertionError):
            np.createNetworkOfPopulation(progression.name, population.name, commutes.name, infectionMatrix.name)


def test_createNetworkOfPopulation_region_mismatch():
    with \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as progression, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as population, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as commutes, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as infectionMatrix:
        progression.write("age,src,dst,rate\n70+,E,E,1.0")
        progression.flush()
        population.write('Health_Board,Sex,Age,Total\nS08000016,Female,70+,31950')
        population.flush()
        commutes.write("S08000015,S08000015,100777")
        commutes.flush()
        infectionMatrix.write(",70+\n70+,1.0")
        infectionMatrix.flush()

        with pytest.raises(AssertionError):
            np.createNetworkOfPopulation(progression.name, population.name, commutes.name, infectionMatrix.name)


def test_createNetworkOfPopulation_infection_matrix_internal_mismatch():
    with \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as progression, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as population, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as commutes, \
      tempfile.NamedTemporaryFile(mode="w+", delete=False) as infectionMatrix:
        progression.write('age,src,dst,rate\n70+,E,E,1.0\n"[10,30)",E,E,1.0')
        progression.flush()
        population.write('Health_Board,Sex,Age,Total\nS08000016,Female,70+,31950\nS08000016,Female,"[10,30)",31950')
        population.flush()
        commutes.write("S08000016,S08000016,100777")
        commutes.flush()
        infectionMatrix.write(',70+\n70+,1.0\n"[10,30),1.0')
        infectionMatrix.flush()

        with pytest.raises(AssertionError):
            np.createNetworkOfPopulation(progression.name, population.name, commutes.name, infectionMatrix.name)


def test_getAges_multiple_ages():
    assert np.getAges({("[0,17)", "S"): 10, ("70+", "S"): 10}) == {"[0,17)", "70+"}


def test_getAges_repeated_ages():
    assert np.getAges({("[0,17)", "S"): 10, ("[0,17)", "S"): 10}) == {"[0,17)"}


def test_getAges_empty():
    assert np.getAges({}) == set()


@pytest.mark.parametrize("progression,exposed,currentState", [
    ({}, {"region1": {}}, {"region1": {}}),
    ({"region1": {}}, {}, {"region1": {}}),
    ({"region1": {}}, {"region1": {}}, {}),
])
def test_createNextStep_region_mismatch_raises_assert_error(progression, exposed, currentState):
    with pytest.raises(AssertionError):
        np.createNextStep(progression, exposed, currentState)
