import copy
import random

import networkx as nx
import pandas as pd
import pytest

from simple_network_sim import network_of_populations as np


def _count_people_per_region(state):
    return [sum(region.values()) for region in state.values()]


@pytest.mark.parametrize("region", ["S08000024", "S08000030"])
@pytest.mark.parametrize("num_infected", [0, 10])
def test_basicSimulationInternalAgeStructure_invariants(data_api, region, num_infected):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        data_api.read_table("human/population"),
        data_api.read_table("human/commutes"),
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        data_api.read_table("human/infection-probability"),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
    )

    initial_population = sum(_count_people_per_region(network.initialState))
    old_network = copy.deepcopy(network)

    result = np.basicSimulationInternalAgeStructure(network, 50, {region: {"[0,17)": num_infected}})

    # population remains constant
    populations = result.groupby("time").total.sum()
    assert all([total == pytest.approx(initial_population) for node, total in populations.to_dict().items()])

    # the graph is unchanged
    assert nx.is_isomorphic(old_network.graph, network.graph)

    # infection matrix is unchanged
    assert list(network.mixingMatrix) == list(old_network.mixingMatrix)
    for a in network.mixingMatrix:
        assert list(network.mixingMatrix[a]) == list(old_network.mixingMatrix[a])
        for b in network.mixingMatrix[a]:
            assert network.mixingMatrix[a][b] == old_network.mixingMatrix[a][b]


@pytest.mark.parametrize("region", ["S08000024", "S08000030", "S08000016"])
@pytest.mark.parametrize("num_infected", [0, 10, 1000])
def test_basicSimulationInternalAgeStructure_no_movement_of_people_invariants(data_api, region, num_infected):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        data_api.read_table("human/population"),
        data_api.read_table("human/commutes"),
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        data_api.read_table("human/infection-probability"),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
        pd.DataFrame([{"Time": 0, "Movement_Multiplier": 0.0, "Contact_Multiplier": 1.0}]),
    )

    initial_population = sum(_count_people_per_region(network.initialState))
    old_network = copy.deepcopy(network)

    result = np.basicSimulationInternalAgeStructure(network, 50, {region: {"[0,17)": num_infected}})

    # population remains constant
    populations = result.groupby("time").total.sum()
    assert all([total == pytest.approx(initial_population) for node, total in populations.to_dict().items()])

    # the graph is unchanged
    assert nx.is_isomorphic(old_network.graph, network.graph)

    # infection matrix is unchanged
    assert list(network.mixingMatrix) == list(old_network.mixingMatrix)
    for a in network.mixingMatrix:
        assert list(network.mixingMatrix[a]) == list(old_network.mixingMatrix[a])
        for b in network.mixingMatrix[a]:
            assert network.mixingMatrix[a][b] == old_network.mixingMatrix[a][b]

    # no spread across regions
    assert result[(result.node != region) & (result.state.isin(network.infectiousStates))].total.sum() == 0.0


@pytest.mark.parametrize("num_infected", [0, 10, 1000])
def test_basicSimulationInternalAgeStructure_no_node_infection_invariant(data_api, num_infected):
    nodes = pd.DataFrame([{"source": "S08000016", "target": "S08000016", "weight": 0.0, "delta_adjustment": 1.0}])
    population = pd.DataFrame([
        {"Health_Board": "S08000016", "Sex": "Female", "Age": "[0,17)", "Total": 31950},
        {"Health_Board": "S08000016", "Sex": "Female", "Age": "[17,70)", "Total": 31950},
        {"Health_Board": "S08000016", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    dampening = pd.DataFrame([{"Time": 0, "Movement_Multiplier": 1.0, "Contact_Multiplier": 0.0}])
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        population,
        nodes,
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        data_api.read_table("human/infection-probability"),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
        dampening,
    )

    initial_population = sum(_count_people_per_region(network.initialState))

    result = np.basicSimulationInternalAgeStructure(network, 50, {"S08000016": {"[17,70)": num_infected}})

    # population remains constant
    populations = result.groupby("time").total.sum()
    assert all([total == pytest.approx(initial_population) for node, total in populations.to_dict().items()])

    # susceptibles are never infected
    for total in result[result.state == "S"].groupby("time").total.sum().to_list():
        assert total == 3 * 31950 - num_infected


def test_basicSimulationInternalAgeStructure_no_infection_prob(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        data_api.read_table("human/population"),
        data_api.read_table("human/commutes"),
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        pd.DataFrame([{"Time": 0, "Value": 0.0}]),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
    )
    susceptibles = 0.0
    for region in network.initialState.values():
        for (age, state) in region.keys():
            if state == "S":
                susceptibles += region[(age, state)]

    people_to_infect = 30
    result = np.basicSimulationInternalAgeStructure(network, 50, {"S08000024": {"[0,17)": people_to_infect}})

    new_susceptibles = result[(result.time == result.time.max()) & (result.state == "S")].total.sum()
    assert new_susceptibles + people_to_infect == susceptibles


def test_basicSimulationInternalAgeStructure_no_infection_prob_before_time_25(data_api):
    def count_susceptibles(state):
        susceptibles = 0.0
        for region in state.values():
            for (age, state) in region.keys():
                if state == "S":
                    susceptibles += region[(age, state)]
        return susceptibles

    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        data_api.read_table("human/population"),
        data_api.read_table("human/commutes"),
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        pd.DataFrame([{"Time": 0, "Value": 0.0}, {"Time": 25, "Value": 1.0}]),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
    )
    people_to_infect = 30
    susceptibles = count_susceptibles(network.initialState) - people_to_infect

    result = np.basicSimulationInternalAgeStructure(network, 50, {"S08000024": {"[0,17)": people_to_infect}})

    # no infection before time 25
    for total in result[(result.time < 25) & (result.state == "S")].groupby("time").total.sum().to_list():
        assert total == susceptibles

    # infections happen after time 25
    for total in result[(result.time >= 25) & (result.state == "S")].groupby("time").total.sum().to_list():
        assert total != susceptibles


def test_internalStateDiseaseUpdate_one_transition():
    current_state = {("o", "E"): 100.0, ("o", "A"): 0.0}
    probs = {"o": {"E": {"A": 0.4, "E": 0.6}, "A": {"A": 1.0}}}

    new_state = np.internalStateDiseaseUpdate(current_state, probs, False, None)

    assert new_state == {("o", "E"): 60.0, ("o", "A"): 40.0}


def test_internalStateDiseaseUpdate_no_transitions():
    current_state = {("o", "E"): 100.0, ("o", "A"): 0.0}
    probs = {"o": {"E": {"E": 1.0}, "A": {"A": 1.0}}}

    new_state = np.internalStateDiseaseUpdate(current_state, probs, False, None)

    assert new_state == {("o", "E"): 100.0, ("o", "A"): 0.0}


def test_doInternalProgressionAllNodes_e_to_a_progression():
    states = {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}}
    probs = {"o": {"E": {"A": 0.4, "E": 0.6}, "A": {"A": 1.0}}}

    progression = np.getInternalProgressionAllNodes(states, probs, False, None)

    assert progression == {"region1": {("o", "E"): 60.0, ("o", "A"): 40.0}}
    assert states == {"region1": {("o", "E"): 100.0, ("o", "A"): 0.0}}  # unchanged


@pytest.mark.parametrize("susceptible", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("infectious", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("asymptomatic", [0.5, 100.0, 300.0])
@pytest.mark.parametrize("contact_rate", [0.0, 0.2, 1.0, 3.0])
@pytest.mark.parametrize("dampening", [1.0, 0.0, 0.5, 2.0])
def test_doInternalInfectionProcess_simple(susceptible, infectious, asymptomatic, contact_rate, dampening):
    current_state = {("m", "S"): susceptible, ("m", "A"): asymptomatic, ("m", "I"): infectious}
    age_matrix = {"m": {"m": contact_rate}}

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, dampening, ["I", "A"], False, None)

    probability_of_susceptible = susceptible / (susceptible + infectious + asymptomatic)
    contacts = contact_rate * (asymptomatic + infectious)
    assert new_infected["m"] == probability_of_susceptible * contacts * dampening


def test_doInternalInfectionProcess_empty_age_group():
    current_state = {("m", "S"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    age_matrix = {"m": {"m": 0.0}}

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, 1.0, ["I", "A"], False, None)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_contact():
    current_state = {("m", "S"): 500.0, ("m", "A"): 100.0, ("m", "I"): 100.0}
    age_matrix = {"m": {"m": 0.0}}

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, 1.0, ["I", "A"], False, None)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_susceptibles():
    current_state = {("m", "S"): 0.0, ("m", "A"): 100.0, ("m", "I"): 100.0}
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, 1.0, ["I", "A"], False, None)

    assert new_infected["m"] == 0.0


def test_doInternalInfectionProcess_no_infectious():
    current_state = {("m", "S"): 300.0, ("m", "A"): 0.0, ("m", "I"): 0.0}
    age_matrix = {"m": {"m": 0.2}}

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, 1.0, ["I", "A"], False, None)

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

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, 1.0, ["I", "A"], False, None)

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

    new_infected = np.getInternalInfectiousContactsInNode(current_state, age_matrix, 1.0, ["I", "A"], False, None)

    assert new_infected["m"] == (20.0 / 470.0) * ((450.0 * 0.2) + (300.0 * 0.5))
    assert new_infected["o"] == (15.0 / 315.0) * ((300.0 * 0.3) + (450.0 * 0.5))


def test_doInternalInfectionProcessAllNodes_single_compartment():
    nodes = {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}}
    age_matrix = {"m": {"m": 0.2}}

    infections = np.getInternalInfectiousContacts(nodes, age_matrix, 1.0, ["I", "A"], False, None)

    assert infections == {"region1": {"m": (300.0 / 400.0) * (0.2 * 100.0)}}
    assert nodes == {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}}  # unchanged


def test_doInternalInfectionProcessAllNodes_large_num_infected_ignored():
    nodes = {"region1": {("m", "S"): 300.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0}}
    age_matrix = {"m": {"m": 5.0}}

    new_infected = np.getInternalInfectiousContacts(nodes, age_matrix, 1.0, ["I", "A"], False, None)

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

    totalIncomingInfectionsByNode = np.getIncomingInfectiousContactsByNode(graph, state, 1.0, ["I", "A"], False, None)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 0.0}


def test_doIncomingInfectionsByNode_no_connections():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")

    state = {
        "r1": {("m", "S"): 100.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 100.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 5.0},
    }

    totalIncomingInfectionsByNode = np.getIncomingInfectiousContactsByNode(graph, state, 1.0, ["I", "A"], False, None)

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

    totalIncomingInfectionsByNode = np.getIncomingInfectiousContactsByNode(graph, state, 1.0, ["I", "A"], False, None)

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

    totalIncomingInfectionsByNode = np.getIncomingInfectiousContactsByNode(graph, state, 1.0, ["I", "A"], False, None)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 0.5 * 0.1 * 0.8}


def test_doIncomingInfectionsByNode_weight_delta_adjustment():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=10, delta_adjustment=0.75)

    state = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }

    totalIncomingInfectionsByNode = np.getIncomingInfectiousContactsByNode(graph, state, 0.5, ["I", "A"], False, None)

    weight = 10 - (5 * 0.75)
    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": weight * 0.1 * 0.8}


def test_doIncomingInfectionsByNode_weight_multiplier():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=10, delta_adjustment=1.0)

    state = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }

    totalIncomingInfectionsByNode = np.getIncomingInfectiousContactsByNode(graph, state, 0.3, ["I", "A"], False, None)

    assert totalIncomingInfectionsByNode == {"r1": 0.0, "r2": 10 * 0.3 * 0.1 * 0.8}


def test_doBetweenInfectionAgeStructured():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=0.5)

    nodes = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }
    original_states = copy.deepcopy(nodes)

    num_infections = np.getExternalInfectiousContacts(graph, nodes, 1.0, ["I", "A"], False, None)

    assert num_infections == {"r1": {"m": 0.0}, "r2": {"m": 0.5 * 0.1 * 0.8}}
    assert nodes == original_states


def test_doBetweenInfectionAgeStructured_multiplier():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=15)

    nodes = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }
    original_states = copy.deepcopy(nodes)

    num_infections = np.getExternalInfectiousContacts(graph, nodes, 0.3, ["I", "A"], False, None)

    assert num_infections == {"r1": {"m": 0.0}, "r2": {"m": 15 * 0.3 * 0.1 * 0.8}}
    assert nodes == original_states


def test_doBetweenInfectionAgeStructured_delta_adjustment():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=15, delta_adjustment=0.3)

    states = {
        "r1": {("m", "S"): 90.0, ("m", "E"): 0.0, ("m", "A"): 5.0, ("m", "I"): 5.0},
        "r2": {("m", "S"): 80.0, ("m", "E"): 0.0, ("m", "A"): 10.0, ("m", "I"): 10.0},
    }
    original_states = copy.deepcopy(states)

    num_infections = np.getExternalInfectiousContacts(graph, states, 0.5, ["I", "A"], False, None)

    delta = 15 - (15 * 0.5)
    weight = 15 - (delta * 0.3)

    assert num_infections == {"r1": {"m": 0.0}, "r2": {"m": weight * 0.1 * 0.8}}
    assert states == original_states


def test_doBetweenInfectionAgeStructured_caps_number_of_infections():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=60)

    nodes = {
        "r1": {("m", "S"): 0.0, ("m", "E"): 0.0, ("m", "A"): 100.0, ("m", "I"): 0.0},
        "r2": {("m", "S"): 30.0, ("m", "E"): 0.0, ("m", "A"): 0.0, ("m", "I"): 0.0},
    }
    original_states = copy.deepcopy(nodes)

    new_infections = np.getExternalInfectiousContacts(graph, nodes, 1.0, ["I", "A"], False, None)

    assert new_infections == {"r1": {"m": 0.0}, "r2": {"m": 30.0}}
    assert nodes == original_states


def test_distributeInfections_cap_infections():
    state = {("m", "S"): 20.0}

    infections = np.distributeContactsOverAges(state, 100, False, None)

    assert infections == {"m": 20.0}


def test_distributeInfections_single_age_always_gets_full_infections():
    state = {("m", "S"): 20.0}

    infections = np.distributeContactsOverAges(state, 10, False, None)

    assert infections == {"m": 10.0}


def test_distributeInfections_infect_proportional_to_susceptibles_in_age_group():
    state = {("m", "S"): 20.0, ("o", "S"): 30.0, ("y", "S"): 40.0}

    infections = np.distributeContactsOverAges(state, 60, False, None)

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

    exposed_state = np.createExposedRegions({"region1": {"m": 5.0, "o": 5.0}}, state)

    assert exposed_state == {"region1": {("m", "S"): 10.0, ("m", "E"): 5.0, ("o", "S"): 5.0, ("o", "E"): 5.0}}


def test_exposeRegion_requires_probabilities_fails_if_age_group_does_not_exist():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0}}

    with pytest.raises(KeyError):
        np.createExposedRegions({"region1": {"m": 10.0, "o": 0.0}}, state)


def test_exposeRegion_multiple_regions():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0}, "region2": {("m", "S"): 15.0, ("m", "E"): 0.0}}

    exposed_state = np.createExposedRegions({"region1": {"m": 10.0}, "region2": {"m": 10.0}}, state)

    assert exposed_state == {
        "region1": {("m", "S"): 5.0, ("m", "E"): 10.0},
        "region2": {("m", "S"): 5.0, ("m", "E"): 10.0}
    }


def test_exposeRegion_only_desired_region():
    state = {"region1": {("m", "S"): 15.0, ("m", "E"): 0.0}, "region2": {("m", "S"): 15.0, ("m", "E"): 0.0}}

    exposed_state = np.createExposedRegions({"region1": {"m": 10.0}}, state)

    assert exposed_state == {
        "region1": {("m", "S"): 5.0, ("m", "E"): 10.0},
        "region2": {("m", "S"): 15.0, ("m", "E"): 0.0}
    }


def test_createNetworkOfPopulation(data_api):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        data_api.read_table("human/population"),
        data_api.read_table("human/commutes"),
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        data_api.read_table("human/infection-probability"),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
    )

    assert network.graph
    assert network.mixingMatrix
    assert network.initialState
    assert network.progression
    assert network.movementMultipliers == {}
    assert set(network.infectiousStates) == {"I", "A"}
    assert network.infectionProb == {0: 1.0}
    assert network.initialInfections == {"S08000016": {"[17,70)": 100}}
    assert network.trials == 1


def test_basicSimulationInternalAgeStructure_invalid_compartment(data_api):
    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            data_api.read_table("human/compartment-transition"),
            data_api.read_table("human/population"),
            data_api.read_table("human/commutes"),
            data_api.read_table("human/mixing-matrix"),
            pd.DataFrame([{"Compartment": "INVALID"}]),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


@pytest.mark.parametrize("time,prob", [(0, -0.5), (0, 10.0), (1, 1.0)])
def test_createNetworkOfPopulation_invalid_infection_probability(data_api, time, prob):
    with pytest.raises(ValueError):
        np.createNetworkOfPopulation(
            data_api.read_table("human/compartment-transition"),
            data_api.read_table("human/population"),
            data_api.read_table("human/commutes"),
            data_api.read_table("human/mixing-matrix"),
            data_api.read_table("human/infectious-compartments"),
            pd.DataFrame([{"Time": time, "Value": prob}]),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_age_mismatch_matrix(data_api):
    progression = pd.DataFrame([{"age": "70+", "src": "E", "dst": "E", "rate": 1.0}])
    population = pd.DataFrame([
        {"Health_Board": "S08000015", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S0800001", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "71+", "target": "71+", "mixing": 1.0}])

    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_age_mismatch_matrix_internal(data_api):
    progression = pd.DataFrame([{"age": "70+", "src": "E", "dst": "E", "rate": 1.0}])
    population = pd.DataFrame([
        {"Health_Board": "S08000015", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S0800001", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "71+", "target": "70+", "mixing": 1.0}])

    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_age_mismatch_population(data_api):
    progression = pd.DataFrame([{"age": "70+", "src": "E", "dst": "E", "rate": 1.0}])
    population = pd.DataFrame([
        {"Health_Board": "S08000015", "Sex": "Female", "Age": "71+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S0800001", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "70+", "target": "70+", "mixing": 1.0}])

    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_age_mismatch_progression(data_api):
    progression = pd.DataFrame([{"age": "71+", "src": "E", "dst": "E", "rate": 1.0}])
    population = pd.DataFrame([
        {"Health_Board": "S08000015", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S0800001", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "70+", "target": "70+", "mixing": 1.0}])

    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_region_mismatch(data_api):
    progression = pd.DataFrame([{"age": "70+", "src": "E", "dst": "E", "rate": 1.0}])
    population = pd.DataFrame([
        {"Health_Board": "S08000016", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S08000015", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "70+", "target": "70+", "mixing": 1.0}])
    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_susceptible_in_progression(data_api):
    progression = pd.DataFrame([
        {"age": "70+", "src": "S", "dst": "E", "rate": 0.5},
        {"age": "70+", "src": "S", "dst": "S", "rate": 0.5},
        {"age": "70+", "src": "E", "dst": "E", "rate": 1.0},
    ])
    population = pd.DataFrame([
        {"Health_Board": "S08000016", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S08000015", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "70+", "target": "70+", "mixing": 1.0}])

    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


def test_createNetworkOfPopulation_transition_to_exposed(data_api):
    progression = pd.DataFrame([
        {"age": "70+", "src": "A", "dst": "E", "rate": 0.7},
        {"age": "70+", "src": "A", "dst": "A", "rate": 0.3},
        {"age": "70+", "src": "E", "dst": "E", "rate": 1.0},
    ])
    population = pd.DataFrame([
        {"Health_Board": "S08000016", "Sex": "Female", "Age": "70+", "Total": 31950},
    ])
    commutes = pd.DataFrame([
        {"source": "S08000015", "target": "S08000015", "weight": 100777.0, "delta_adjustment": 1.0}
    ])
    mixingMatrix = pd.DataFrame([{"source": "70+", "target": "70+", "mixing": 1.0}])

    with pytest.raises(AssertionError):
        np.createNetworkOfPopulation(
            progression,
            population,
            commutes,
            mixingMatrix,
            data_api.read_table("human/infectious-compartments"),
            data_api.read_table("human/infection-probability"),
            data_api.read_table("human/initial-infections"),
            data_api.read_table("human/trials"),
        )


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
        np.createNextStep(progression, exposed, currentState, 1.0, False, None)


def test_createNextStep_keep_susceptibles():
    currState = {"r1": {("70+", "S"): 30.0, ("70+", "E"): 20.0}}

    nextStep = np.createNextStep({"r1": {}}, {"r1": {}}, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 30.0, ("70+", "E"): 0.0}}


def test_createNextStep_update_infection():
    currState = {"r1": {("70+", "S"): 30.0, ("70+", "E"): 0.0}}
    progression = {"r1": {}}
    exposed = {"r1": {"70+": 20.0}}

    nextStep = np.createNextStep(progression, exposed, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 15.22846460770688, ("70+", "E"): 14.77153539229312}}


def test_createNextStep_use_infection_rate():
    currState = {"r1": {("70+", "S"): 30.0, ("70+", "E"): 0.0}}
    progression = {"r1": {}}
    exposed = {"r1": {"70+": 20.0}}

    nextStep = np.createNextStep(progression, exposed, currState, 0.5, False, None)

    assert nextStep == {"r1": {("70+", "S"): 22.61423230385344, ("70+", "E"): 7.38576769614656}}


def test_createNextStep_susceptible_in_progression():
    currState = {"r1": {("70+", "S"): 30.0}}
    progression = {"r1": {("70+", "S"): 7.0}}
    exposed = {"r1": {}}

    with pytest.raises(AssertionError):
        np.createNextStep(progression, exposed, currState, 1.0, False, None)


def test_createNextStep_progression_nodes():
    currState = {"r1": {("70+", "S"): 30.0, ("70+", "E"): 10.0}}
    progression = {"r1": {("70+", "E"): 7.0, ("70+", "A"): 3.0}}
    exposed = {"r1": {"70+": 10.0}}

    nextStep = np.createNextStep(progression, exposed, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 21.374141812742014, ("70+", "E"): 15.625858187257986, ("70+", "A"): 3.0}}


def test_createNextStep_very_small_susceptible():
    currState = {"r1": {("70+", "S"): 0.7, ("70+", "E"): 0.0}}
    progression = {"r1": {}}
    exposed = {"r1": {"70+": 0.5}}

    nextStep = np.createNextStep(progression, exposed, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 0.19999999999999996, ("70+", "E"): 0.5}}


def test_createNextStep_zero_susceptible():
    currState = {"r1": {("70+", "S"): 0., ("70+", "E"): 0.0}}
    progression = {"r1": {}}
    exposed = {"r1": {"70+": 0.}}

    nextStep = np.createNextStep(progression, exposed, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 0., ("70+", "E"): 0.}}


def test_createNextStep_susceptible_smaller_than_exposed():
    currState = {"r1": {("70+", "S"): 10., ("70+", "E"): 10.}}
    progression = {"r1": {}}
    exposed = {"r1": {"70+": 15.}}
    nextStep = np.createNextStep(progression, exposed, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 2.058911320946491, ("70+", "E"): 7.941088679053509}}

    currState = {"r1": {("70+", "S"): 0.5, ("70+", "E"): 0.}}
    progression = {"r1": {}}
    exposed = {"r1": {"70+": 0.75}}
    nextStep = np.createNextStep(progression, exposed, currState, 1.0, False, None)

    assert nextStep == {"r1": {("70+", "S"): 0., ("70+", "E"): 0.5}}


def test_getSusceptibles():
    states = {("70+", "S"): 10, ("70+", "E"): 20, ("[17,70)", "S"): 15}

    assert np.getSusceptibles("70+", states) == 10


def test_getSusceptibles_non_existent():
    states = {}

    with pytest.raises(KeyError):
        np.getSusceptibles("75+", states)


def test_getInfectious():
    states = {("70+", "S"): 10, ("70+", "E"): 20, ("70+", "I"): 7, ("70+", "A"): 11, ("[17,70)", "S"): 15}

    assert np.getInfectious("70+", states, ["I", "A"]) == 18


def test_getInfectious_with_a2():
    states = {("70+", "S"): 10, ("70+", "E"): 20, ("70+", "I"): 7,
              ("70+", "A"): 11, ("70+", "A2"): 5, ("[17,70)", "S"): 15}

    assert np.getInfectious("70+", states, ["I", "A", "A2"]) == 23.0


def test_getInfectious_empty():
    states = {("70+", "S"): 10, ("70+", "E"): 20, ("70+", "I"): 7, ("70+", "A"): 11, ("[17,70)", "S"): 15}

    assert np.getInfectious("70+", states, []) == 0.0


def test_getInfectious_invalid_state():
    states = {("70+", "S"): 10, ("70+", "E"): 20, ("70+", "I"): 7, ("70+", "A"): 11, ("[17,70)", "S"): 15}

    with pytest.raises(KeyError):
        np.getInfectious("70+", states, ["X"])


def test_getInfectious_non_existant():
    states = {}

    with pytest.raises(KeyError):
        np.getInfectious("70+", states, ["I", "A"])


@pytest.mark.parametrize(
    "delta_adjustment,multiplier",
    [(1.0, 1.0), (0.7, 1.0), (1.0, 0.7), (0.0, 0.8), (0.8, 0.0), (2.0, 1.0), (2.0, 2.0), (0.7, 2.0)]
)
def test_getWeight(delta_adjustment, multiplier):
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=30.0, delta_adjustment=delta_adjustment)

    assert np.getWeight(graph, "r1", "r2", multiplier) == 30.0 - delta_adjustment * (1 - multiplier) * 30.0


def test_getWeight_no_delta():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=100.0, delta_adjustment=0.7)

    assert np.getWeight(graph, "r1", "r2", 1.0) == 100.0


def test_getWeight_no_delta_adjustment():
    graph = nx.DiGraph()
    graph.add_node("r1")
    graph.add_node("r2")
    graph.add_edge("r1", "r2", weight=100.0)

    assert np.getWeight(graph, "r1", "r2", 0.5) == 50.0


@pytest.mark.parametrize("regions", [2, 4])
@pytest.mark.parametrize("age_groups", [['70+']])
@pytest.mark.parametrize("infected", [100, 10])
def test_randomlyInfectRegions(data_api, regions, age_groups, infected):
    network = np.createNetworkOfPopulation(
        data_api.read_table("human/compartment-transition"),
        data_api.read_table("human/population"),
        data_api.read_table("human/commutes"),
        data_api.read_table("human/mixing-matrix"),
        data_api.read_table("human/infectious-compartments"),
        data_api.read_table("human/infection-probability"),
        data_api.read_table("human/initial-infections"),
        data_api.read_table("human/trials"),
    )

    random.seed(3)
    infections = np.randomlyInfectRegions(network, regions, age_groups, infected)

    assert len(infections) == regions
    assert list(age_groups[0] in infection for infection in infections.values())
    assert all(infection[age_groups[0]] == infected for infection in infections.values())
