"""
This package implements the network of populations simuation. This models regions as nodes with intra node rules and
different inter node transmission rules. The end result of the model is a timeseries of the number of people in each
node, compartment and age.
"""
# pylint: disable=import-error
# pylint: disable=too-many-lines
import logging
import random
import copy
import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, NamedTuple, List, Optional, cast

import networkx as nx
import pandas as pd

from simple_network_sim import loaders
from simple_network_sim.common import Lazy

logger = logging.getLogger(__name__)

# Type aliases used to make the types for the functions below easier to read
Age = str
Compartment = str
NodeName = str
Time = int

RESULT_DTYPES = {"time": "int16", "age": "category", "state": "category", "node": "category"}


class NetworkOfPopulation(NamedTuple):
    """
    This type has all the internal data used by this model
    """
    progression: Dict[Age, Dict[Compartment, Dict[Compartment, float]]]
    initialState: Dict[NodeName, Dict[Tuple[Age, Compartment], float]]
    graph: nx.DiGraph
    mixingMatrix: loaders.MixingMatrix
    movementMultipliers: Dict[Time, loaders.Multiplier]
    infectiousStates: List[Compartment]
    infectionProb: Dict[Time, float]
    initialInfections: Dict[NodeName, Dict[Age, float]]
    trials: int
    stochastic: bool
    randomState: np.random.Generator


# CurrentlyInUse
def basicSimulationInternalAgeStructure(
    network: NetworkOfPopulation,
    timeHorizon: int,
    initialInfections: Dict[NodeName, Dict[Age, float]]
) -> pd.DataFrame:
    """Run the simulation of a disease progressing through a network of regions.

    :param network: This is a NetworkOfPopulation instance which will have the states field modified by this function.
    :param timeHorizon: How many times to run the simulation. Each new time means a new entry to the network.states dict
    :param initialInfections: Initial infections of the disease
    :return: A time series of the size of the infectious population.
    """
    history = []

    multipliers = network.movementMultipliers.get(0, loaders.Multiplier(contact=1.0, movement=1.0))
    infectionProb = network.infectionProb[0]  # no default value, time zero must exist

    current = createExposedRegions(initialInfections, network.initialState)
    df = nodesToPandas(0, current)
    logger.debug("Time (0/%s). Status: %s", timeHorizon, Lazy(lambda: df.groupby("state").total.sum().to_dict()))
    history.append(df)
    for time in range(timeHorizon):
        # we are building the interactions for time + 1, so that's the multiplier value we need to use
        multipliers = network.movementMultipliers.get(time + 1, multipliers)
        infectionProb = network.infectionProb.get(time + 1, infectionProb)

        progression = getInternalProgressionAllNodes(
            current,
            network.progression,
            network.stochastic,
            network.randomState
        )

        internalContacts = getInternalInfectiousContacts(
            current,
            network.mixingMatrix,
            multipliers.contact,
            network.infectiousStates,
            network.stochastic,
            network.randomState
        )
        externalContacts = getExternalInfectiousContacts(
            network.graph,
            current,
            multipliers.movement,
            network.infectiousStates,
            network.stochastic,
            network.randomState
        )
        contacts = mergeContacts(internalContacts, externalContacts)

        current = createNextStep(
            progression,
            contacts,
            current,
            infectionProb,
            network.stochastic,
            network.randomState
        )

        df = nodesToPandas(time + 1, current)
        logger.debug(
            "Time (%s/%s). Status: %s",
            time + 1,
            timeHorizon,
            Lazy(lambda: df.groupby("state").total.sum().to_dict())
        )
        history.append(df)

    return pd.concat(history, copy=False, ignore_index=True)


def nodesToPandas(time: int, nodes: Dict[NodeName, Dict[Tuple[Age, Compartment], float]]) -> pd.DataFrame:
    """
    Converts a dict of nodes into a pandas DataFrame

    >>> nodesToPandas(0, {"nodea": {("70+", "S"): 10.0, ("70+", "E"): 15.0}, "nodeb": {("[17,70", "S"): 15.0}})  # doctest: +NORMALIZE_WHITESPACE
        time   node     age state  total
    0     0  nodea     70+     S   10.0
    1     0  nodea     70+     E   15.0
    2     0  nodeb  [17,70     S   15.0
    >>> nodesToPandas(0, {})  # doctest: +NORMALIZE_WHITESPACE
    Empty DataFrame
    Columns: [time, node, age, state, total]
    Index: []

    :param time: time that will be inserted into every row
    :param nodes: a dict of nodes
    :return: a pandas dataframe representation of the nodes
    """
    rows = []
    for name, node in nodes.items():
        for (age, state), value in node.items():
            rows.append([time, name, age, state, value])
    return pd.DataFrame(rows, columns=["time", "node", "age", "state", "total"]).astype(RESULT_DTYPES, copy=True)


def totalIndividuals(nodeState: Dict[Tuple[Age, Compartment], float]) -> float:
    """This function takes a node (region) and counts individuals from every age and state.

    :param nodeState: The disease status of a node (or region) stratified by age.
    :return: The total number of individuals.
    """
    return sum(nodeState.values())


def getAges(node: Dict[Tuple[Age, Compartment], float]) -> List[Age]:
    """Get the set of ages from the node.

    :param node: The disease states of the population stratified by age.
    :return: The unique collection of ages.
    """
    ages = set()
    for (age, state) in node:
        ages.add(age)
    return sorted(list(ages))


def getTotalInAge(nodeState: Dict[Tuple[Age, Compartment], float], ageTest: Age):
    """Get the size of the population within an age group.

    :param nodeState: The disease states of the population stratified by age.
    :param ageTest: The age range of the population.
    :return: The population size within the age range.
    """
    total = 0
    for (age, _), value in nodeState.items():
        if age == ageTest:
            total += value
    return total


def getTotalInfectious(node: Dict[Tuple[Age, Compartment], float], infectiousStates: List[Compartment]) -> float:
    """Get the total number of infectious individuals regardless of age in the node.

    :param node: The disease status of the population stratified by age.
    :param infectiousStates: States that are considered infectious
    :return: The total number of infectious individuals.
    """
    total = 0.0
    for (_, compartment), value in node.items():
        if compartment in infectiousStates:
            total += value
    return total


def getTotalSuscept(nodeState: Dict[Tuple[Age, Compartment], float]) -> float:
    """Get the total number of susceptible individuals regardless of age in the node

    :param nodeState: The disease status of the population stratified by age.
    :return: The total number of susceptible individuals.
    """
    totalSusHere = 0.0
    for age in getAges(nodeState):
        totalSusHere += getSusceptibles(age, nodeState)
    return totalSusHere


def distributeContactsOverAges(
        nodeState: Dict[Tuple[Age, Compartment], float],
        newContacts: float,
        stochastic: bool,
        random_state: np.random.Generator,
) -> Dict[Age, float]:
    """Distribute the number of new contacts across a region. This function distributes
    newContacts among the susceptibles according to the relative size of each age group. Note:
    fractional people will come out of this.

    :param nodeState: The disease status of the population stratified by age.
    :param newContacts: The number of new contacts to be distributed across age ranges.
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of new infections in each age group.
    """
    ageToSus = {}
    totalSus = 0.0
    for age in getAges(nodeState):
        sus = getSusceptibles(age, nodeState)
        ageToSus[age] = sus
        totalSus += sus

    if totalSus < newContacts:
        logger.error("totalSus < incoming contacts (%s < %s) - adjusting to totalSus", totalSus, newContacts)
        newContacts = totalSus

    if totalSus > 0:
        newInfectionsByAge = _distributeContactsOverAges(ageToSus, totalSus, newContacts, stochastic, random_state)
    else:
        newInfectionsByAge = {age: 0 for age in ageToSus}

    return newInfectionsByAge


def _distributeContactsOverAges(
        ageToSusceptibles: Dict[Age, float],
        totalSusceptibles: float,
        newContacts: float,
        stochastic: bool,
        random_state: np.random.Generator
) -> Dict[Age, float]:
    """Distribute the number of new contacts across a region. This function distributes
    newContacts among the susceptibles according to the relative size of each age group.
    Two versions are possible:
    1) Deterministic
    We simply allocate newContacts by proportion of people in each age groups
    2) Stochastic
    newContacts are sampled randomly from the population from a Multinomial distribution,
    with: k=newContacts, p=proportion of people in age groups

    :param ageToSusceptibles: The number of susceptibles in each age group
    :param totalSusceptibles: Total number of susceptible people in all age groups
    :param newContacts: The number of new contacts to be distributed across age ranges.
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of new infections in each age group.
    """
    if stochastic:
        assert isinstance(newContacts, int) or newContacts.is_integer()

        ageProbabilities = np.array(list(ageToSusceptibles.values())) / totalSusceptibles
        allocationsByAge = stats.multinomial.rvs(newContacts, ageProbabilities, random_state=random_state)

        return dict(zip(ageToSusceptibles.keys(), allocationsByAge))
    else:
        return {age: (sus / totalSusceptibles) * newContacts for age, sus in ageToSusceptibles.items()}


def getIncomingInfectiousContactsByNode(
        graph: nx.DiGraph,
        currentState: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
        movementMultiplier: float,
        infectiousStates: List[Compartment],
        stochastic: bool,
        random_state: np.random.Generator,
) -> Dict[NodeName, float]:
    """Determine the number of new infections at each node of a graph based on incoming people
    from neighbouring nodes.

    :param graph: A graph with each region as a node and the weights corresponding to the movements
                  between regions. Edges must contain weight and delta_adjustment attributes (assumed 1.0)
    :param currentState: The current state for every region.
    :param movementMultiplier: a multiplier applied to each edge (movement) in the network.
    :param infectiousStates: States that are considered infectious
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: the number of new infections in each region.
    """
    infectiousByNode: Dict[NodeName, float] = {}
    totalByNode: Dict[NodeName, float] = {}
    # Precompute this so that we avoid expensive calls inside the O(n^2) part of the algorithm as most as we can
    for name, node in currentState.items():
        infectiousByNode[name] = getTotalInfectious(node, infectiousStates)
        totalByNode[name] = totalIndividuals(node)
    contactsByNode: Dict[NodeName, float] = {}

    # Surprisingly, iterating over graph.edges is actually slower than going through the dicts and calling
    # graph.predecessor when needed
    for receivingVertex in currentState:
        totalSusceptHere = getTotalSuscept(currentState[receivingVertex])
        contactsByNode[receivingVertex] = 0

        if totalSusceptHere > 0:
            for givingVertex in graph.predecessors(receivingVertex):
                if givingVertex == receivingVertex:
                    continue
                totalInfectedGiving = infectiousByNode[givingVertex]
                if totalInfectedGiving > 0:
                    weight = getWeight(graph, givingVertex, receivingVertex, movementMultiplier)
                    fractionGivingInfected = totalInfectedGiving / totalByNode[givingVertex]
                    fractionReceivingSus = totalSusceptHere / totalByNode[receivingVertex]

                    contactsByNode[receivingVertex] += _computeInfectiousCommutes(
                        weight,
                        fractionGivingInfected,
                        fractionReceivingSus,
                        stochastic,
                        random_state
                    )

    return contactsByNode


def _computeInfectiousCommutes(
        weight: float,
        fractionGivingInfected: float,
        fractionReceivingSus: float,
        stochastic: bool,
        random_state: np.random.Generator
) -> float:
    """Transforms the weights (commutes) into potentially infectious commutes, that
    originate from infectious people, and target susceptible people. Two modes:
    1) Deterministic
    We multiply the commute by the proportion of infectious people in giving node,
    and by the proportion of susceptible people in receiving age group
    2) Stochastic
    We assume commutes are randomly distributed across people.
    We sample int(weight) from a binomial distribution with p=fractionReceivingSus.
    This gives the number of commutes that target susceptible people. Then we sample
    the result into another binomial distribution with p=fractionGivingInfected. This gives
    the number of commutes that target susceptible people, and originate from infectious
    people.

    :param weight: Raw number of commutes
    :param fractionGivingInfected: Fraction of infectious people in source node of commute
    :param fractionReceivingSus: Fraction of susceptible people in destination node of commute
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: the number of new infections in each region.
    """
    if stochastic:
        # weight can be fractional because of the movement multiplier, round it
        contacts = stats.binom.rvs(int(round(weight)), fractionReceivingSus, random_state=random_state)
        contacts = stats.binom.rvs(contacts, fractionGivingInfected, random_state=random_state)
        return contacts
    else:
        return weight * fractionGivingInfected * fractionReceivingSus


def getWeight(graph: nx.DiGraph, orig: str, dest: str, multiplier: float) -> float:
    """Get the weight of the edge from orig to dest in the graph. This weight is expected to be
    proportional to the movement between nodes. If the edge doesn't have a weight, 1.0 is assumed
    and the returned weight is adjusted by the multiplier and any delta_adjustment on the edge.

    :param graph: A graph with each region as a node and the weights corresponding to the commutes
                  between regions. Edges must contain weight and delta_adjustment attributes (assumed 1.0)
    :param orig: The vertex people are coming from.
    :param dest: The vertex people are going to.
    :param multiplier: Value that will dampen or heighten movements between nodes.
    :return: The final weight value
    """
    edge = graph.get_edge_data(orig, dest)
    if "weight" not in edge:
        logger.error("No weight available for edge %s,%s assuming 1.0", orig, dest)
        weight = 1.0
    else:
        weight = edge["weight"]

    if "delta_adjustment" not in edge:
        logger.error("delta_adjustment not available for edge %s,%s assuming 1.0", orig, dest)
        delta_adjustment = 1.0
    else:
        delta_adjustment = edge["delta_adjustment"]

    delta = weight - (weight * multiplier)
    # The delta_adjustment is applied on the delta. It can either completely cancel any changes (factor = 0.0) or
    # enable it fully (factor = 1.0). If the movement multiplier doesn't make any changes to the node's movements (ie.
    # multiplier = 1.0), then the delta_adjustment will have no effect.
    return weight - (delta * delta_adjustment)


def getExternalInfectiousContacts(
        graph: nx.DiGraph,
        nodes: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
        movementMultiplier: float,
        infectiousStates: List[Compartment],
        stochastic: bool,
        random_state: np.random.Generator,
) -> Dict[NodeName, Dict[Age, float]]:
    """Calculate the number of new infections in each region. The infections are distributed
    proportionally to the number of susceptibles in the destination node and infected in the origin
    node. The infections are distributed to each age group according to the number of susceptible
    people in them.

    :param graph: A graph with each region as a node and the weights corresponding to the movements
                  between regions. Edges must contain weight and delta_adjustment attributes (assumed 1.0)
    :param nodes: The disease status in each region stratified by age.
    :param movementMultiplier: A multiplier applied to each edge (movement between nodes) in the
                               network.
    :param infectiousStates: States that are considered infectious
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of new infections in each region stratified by age.
    """
    infectionsByNode = {}

    incomingContacts = getIncomingInfectiousContactsByNode(
        graph,
        nodes,
        movementMultiplier,
        infectiousStates,
        stochastic,
        random_state
    )

    for name, vertex in incomingContacts.items():
        infectionsByNode[name] = distributeContactsOverAges(nodes[name], vertex, stochastic, random_state)

    return infectionsByNode


def getInternalInfectiousContactsInNode(
    currentInternalStateDict: Dict[Tuple[Age, Compartment], float],
    mixingMatrix: loaders.MixingMatrix,
    contactsMultiplier: float,
    infectiousStates: List[Compartment],
    stochastic: bool,
    random_state: np.random.Generator,
) -> Dict[Age, float]:
    """Calculate the new infections due to mixing within the region and stratify them by age.

    :param currentInternalStateDict: The disease status of the population stratified by age.
    :param mixingMatrix: Stores expected numbers of interactions between people of
                         different ages.
    :param contactsMultiplier: Multiplier applied to the number of infectious contacts.
    :param infectiousStates: States that are considered infectious
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of new infections stratified by age.
    """
    infectiousContacts: Dict[Age, float] = {}
    for ageTo in mixingMatrix:
        infectiousContacts[ageTo] = 0

        susceptibles = getSusceptibles(ageTo, currentInternalStateDict)
        totalInAge = getTotalInAge(currentInternalStateDict, ageTo)

        if susceptibles > 0 and totalInAge > 0.0:
            for ageFrom in mixingMatrix[ageTo]:
                infectious = getInfectious(ageFrom, currentInternalStateDict, infectiousStates)

                infectiousContacts[ageTo] += _computeInfectiousContacts(
                    mixingMatrix[ageTo][ageFrom] * contactsMultiplier,
                    infectious,
                    susceptibles,
                    totalInAge,
                    stochastic,
                    random_state
                )

    return infectiousContacts


def _computeInfectiousContacts(
        contacts: float,
        infectious: float,
        susceptibles: float,
        totalInAge: float,
        stochastic: bool,
        random_state: np.random.Generator,
) -> float:
    """From raw contacts (between any two people in different age groups), filters
    only those contacts that originated from an infectious person and received
    by a susceptible person. The contact did not yet lead to a new infection,
    we need to multiply by the infection probability. Two modes:
    1) Deterministic
    We simply multiply the contacts by number of infectious, and by proportion of susceptibles
    2) Stochastic
    For each infectious individual we sample from a Poisson distribution with mean the number
    of contacts. Then the output k (number of contacts for one particular person), is inputted
    in a Hypergeometric distribution, where we sample k individuals with replacement in the
    population of susceptibles. This provides potentially infectious contacts, which are summed.

    :param contacts: The number of reported contacts
    :param infectious: The number of infectious people from which the contacts
                       can originate
    :param susceptibles: The number of susceptible people to which the contacts
                         can target
    :param totalInAge: Total number of people in target population
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of potentially infectious contacts.
    """
    if stochastic:
        assert isinstance(infectious, int) or infectious.is_integer()
        assert isinstance(totalInAge, int) or totalInAge.is_integer()
        assert isinstance(susceptibles, int) or susceptibles.is_integer()

        numberOfContacts = stats.poisson.rvs(contacts, size=int(infectious), random_state=random_state)
        numberOfContacts = numberOfContacts[numberOfContacts > 0]  # If 0, the rvs calls fails, but mathematically is 0
        numberOfContacts = stats.hypergeom.rvs(int(totalInAge), int(susceptibles), numberOfContacts, random_state=random_state)
        return cast(float, np.sum(numberOfContacts))
    else:
        return infectious * contacts * (susceptibles / totalInAge)


def getInternalInfectiousContacts(
        nodes: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
        mixingMatrix: loaders.MixingMatrix,
        contactsMultiplier: float,
        infectiousStates,
        stochastic: bool,
        random_state: np.random.Generator,
) -> Dict[NodeName, Dict[Age, float]]:
    """Calculate the new infections and stratify them by region and age.

    :param nodes: The disease status in each region stratified by age.
    :param mixingMatrix: Stores expected numbers of interactions between people of different ages.
    :param contactsMultiplier: Multiplier applied to the number of infectious contacts.
    :param infectiousStates: States that are considered infectious
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of exposed in each region stratified by age.
    """
    contacts: Dict[NodeName, Dict[Age, float]] = {}

    for name, node in nodes.items():
        contacts[name] = getInternalInfectiousContactsInNode(
            node,
            mixingMatrix,
            contactsMultiplier,
            infectiousStates,
            stochastic,
            random_state
        )

    return contacts


def internalStateDiseaseUpdate(
        currentInternalStateDict: Dict[Tuple[Age, Compartment], float],
        diseaseProgressionProbs: Dict[Age, Dict[Compartment, Dict[Compartment, float]]],
        stochastic: bool,
        random_state: np.random.Generator,
) -> Dict[Tuple[Age, Compartment], float]:
    """Returns the status of exposed individuals, moving them into the next disease state with a
    probability defined in the given progression matrix.

    :param currentInternalStateDict: The disease status of the population stratified by age.
    :param diseaseProgressionProbs: A matrix with the probabilities if progressing from one state
                                    to the next.
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: the numbers in each exposed state stratified by age
    """
    newStates = {}

    for (age, state), people in currentInternalStateDict.items():
        outTransitions = diseaseProgressionProbs[age].get(state, {})
        _internalStateDiseaseUpdate(age, state, people, outTransitions, newStates, stochastic, random_state)

    return newStates


def _internalStateDiseaseUpdate(
        age: Age,
        state: Compartment,
        people: float,
        outTransitions: Dict[Compartment, float],
        newStates: Dict[Tuple[Age, Compartment], float],
        stochastic: bool,
        random_state: np.random.Generator
):
    """Returns the status of exposed individuals, moving them into the next disease state with a
    probability defined in the given progression matrix. Two modes available:
    1) Deterministic mode
    The people in the given state are transferred into the next available states
    by multiplying by the probability of going to each state.
    2) Stochastic mode
    We sample from a multinomial distribution with parameters: k=people, p=outTransitions.

    :param age: Current age
    :param age: Current state
    :param people: Number of people in current age, state
    :param outTransitions: Transition probabilities out of the current age, state,
                           to next potential ones
    :param newStates: Distribution of people in next states
    :param stochastic: Use the model in stochastic mode?
    :param random_state: Random number generator used for the model
    :return: the numbers in each exposed state stratified by age
    """
    if stochastic:
        if state == SUSCEPTIBLE_STATE:
            return

        assert isinstance(people, int) or people.is_integer()

        outRepartitions = stats.multinomial.rvs(people, list(outTransitions.values()), random_state=random_state)
        outRepartitions = dict(zip(outTransitions.keys(), outRepartitions))

        for nextState, nextStateCases in outRepartitions.items():
            newStates.setdefault((age, nextState), 0.0)
            newStates[(age, nextState)] += nextStateCases
    else:
        for nextState, transition in outTransitions.items():
            newStates.setdefault((age, nextState), 0.0)
            newStates[(age, nextState)] += transition * people


def getInternalProgressionAllNodes(
       currStates: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
       diseaseProgressionProbs: Dict[Age, Dict[Compartment, Dict[Compartment, float]]],
       stochastic: bool,
       random_state: np.random.Generator,
) -> Dict[NodeName, Dict[Tuple[Age, Compartment], float]]:
    """
    Given the size of the population in each exposed state, calculate the numbers that progress
    the next disease state based on the progression matrix.

    :param currStates: The current state for every region is not modified.
    :param diseaseProgressionProbs: A matrix with the probabilities if progressing from one state
                                    to the next.
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The number of individuals that have progressed into each exposed stage, stratified by
             region and age.
    """
    progression = {}
    for regionID, currRegion in currStates.items():
        progression[regionID] = internalStateDiseaseUpdate(
            currRegion,
            diseaseProgressionProbs,
            stochastic,
            random_state
        )

    return progression


def mergeContacts(*args) -> Dict[NodeName, Dict[Age, float]]:
    """From a list of exposed cases stratified by age and region, merge them into a single
    collection

    :param args: A variable list of regional exposure numbers by age.
    :type args: A variable list of dictionaries that have a region as a key and the value is a
                dictionary of {age:number of exposed}.
    :return: A dictionary containing the merged number of exposed.
    """
    exposedTotal = {}
    for infectionsDict in args:
        for regionID, region in infectionsDict.items():
            exposedRegionTotal = exposedTotal.setdefault(regionID, {})
            for age, exposed in region.items():
                exposedRegionTotal[age] = exposedRegionTotal.setdefault(age, 0.0) + exposed
    return exposedTotal


def createExposedRegions(
    infections: Dict[NodeName, Dict[Age, float]],
    states: Dict[NodeName, Dict[Tuple[Age, Compartment], float]]
) -> Dict[NodeName, Dict[Tuple[Age, Compartment], float]]:
    """Creates a new the state of a region, adding the new infections.

    :param infections: The number of infections per region per age
    :param states: The current state for every region is not modified
    :return: A dictionary containing the merged number of exposed
    """
    exposedStates = copy.deepcopy(states)
    for nodeName, node in infections.items():
        for age, value in node.items():
            expose(age, value, exposedStates[nodeName])

    return exposedStates


def getInfectious(
        age: Age,
        currentInternalStateDict: Dict[Tuple[Age, Compartment], float],
        infectiousStates: List[Compartment],
) -> float:
    """Calculate the total number of individuals in infectious states in an age range.

    :param age: The age (range)
    :param currentInternalStateDict: The disease status of the population stratified by age.
    :param infectiousStates: States that are considered infectious
    :return: The number of individuals in an infectious state and age range.
    """
    total = 0.0
    for state in infectiousStates:
        total += currentInternalStateDict[(age, state)]
    return total


# The functions below are the only operations that need to know about the actual state values.
SUSCEPTIBLE_STATE = "S"
EXPOSED_STATE = "E"


def createNetworkOfPopulation(
    compartment_transition_table: pd.DataFrame,
    population_table: pd.DataFrame,
    commutes_table: pd.DataFrame,
    mixing_matrix_table: pd.DataFrame,
    infectious_states: pd.DataFrame,
    infection_prob: pd.DataFrame,
    initial_infections: pd.DataFrame,
    trials: pd.DataFrame,
    movement_multipliers_table: pd.DataFrame = None,
    stochastic_mode: pd.DataFrame = None,
    random_seed: pd.DataFrame = None
) -> NetworkOfPopulation:
    """Create the network of the population, loading data from files.

    :param compartment_transition_table: pd.Dataframe specifying the transition rates between infected compartments.
    :param population_table: pd.Dataframe with the population size in each region by gender and age.
    :param commutes_table: pd.Dataframe with the movements between regions.
    :param mixing_matrix_table: pd.Dataframe with the age infection matrix.
    :param infectious_states: States that are considered infectious
    :param infection_prob: Probability that a given contact will result in an infection
    :param initial_infections: Initial infections of the population at time 0
    :param trials: Number of trials for the model
    :param movement_multipliers_table: pd.Dataframe with the movement multipliers. This may be None, in
                                       which case no multipliers are applied to the movements.
    :param stochastic_mode: Use stochastic mode for the model
    :param random_seed: Random number generator seed used for stochastic mode
    :return: The constructed network
    """
    infection_prob = loaders.readInfectionProbability(infection_prob)

    infectious_states = loaders.readInfectiousStates(infectious_states)
    initial_infections = loaders.readInitialInfections(initial_infections)
    trials = loaders.readTrials(trials)
    stochastic_mode = loaders.readStochasticMode(stochastic_mode)
    random_seed = loaders.readRandomSeed(random_seed)

    # diseases progression matrix
    progression = loaders.readCompartmentRatesByAge(compartment_transition_table)

    # population census data
    population = loaders.readPopulationAgeStructured(population_table)

    # Check some requirements for this particular model to work with the progression matrix
    all_states = set()
    for states in progression.values():
        assert SUSCEPTIBLE_STATE not in states, "progression from susceptible state is not allowed"
        for state, nextStates in states.items():
            for nextState in nextStates:
                all_states.add(state)
                all_states.add(nextState)
                assert state == nextState or nextState != EXPOSED_STATE, \
                    "progression into exposed state is not allowed other than in self reference"
    assert (set(infectious_states) - all_states) == set(), \
        f"mismatched infectious states and states {infectious_states} {all_states}"

    # people movement's graph
    graph = loaders.genGraphFromContactFile(commutes_table)

    # movement multipliers (dampening or heightening)
    if movement_multipliers_table is not None:
        movementMultipliers = loaders.readMovementMultipliers(movement_multipliers_table)
    else:
        movementMultipliers: Dict[int, loaders.Multiplier] = {}

    # age-based infection matrix
    mixingMatrix = loaders.MixingMatrix(mixing_matrix_table)

    agesInInfectionMatrix = set(mixingMatrix)
    for age in mixingMatrix:
        assert agesInInfectionMatrix == set(mixingMatrix[age]), "infection matrix columns/rows mismatch"

    # Checks across datasets
    assert agesInInfectionMatrix == set(progression.keys()), "infection matrix and progression ages mismatch"
    assert agesInInfectionMatrix == {age for region in population.values() for age in region}, \
        "infection matrix and population ages mismatch"
    disconnected_nodes = set(population.keys()) - set(graph.nodes())
    if disconnected_nodes:
        logger.warning("These nodes have no contacts in the current network: %s", disconnected_nodes)

    state0: Dict[str, Dict[Tuple[str, str], float]] = {}
    for node in list(graph.nodes()):
        region = state0.setdefault(node, {})
        for age, compartments in progression.items():
            if node not in population:
                logger.warning("Node %s is not in the population table, assuming population of 0 for all ages", node)
                pop = 0.0
            else:
                pop = population[node][age]
            region[(age, SUSCEPTIBLE_STATE)] = pop
            for compartment in compartments:
                region[(age, compartment)] = 0

    logger.info("Nodes: %s, Ages: %s, States: %s", len(state0), agesInInfectionMatrix, all_states)
    return NetworkOfPopulation(
        progression=progression,
        graph=graph,
        initialState=state0,
        mixingMatrix=mixingMatrix,
        movementMultipliers=movementMultipliers,
        infectiousStates=infectious_states,
        infectionProb=infection_prob,
        initialInfections=initial_infections,
        trials=trials,
        stochastic=stochastic_mode,
        randomState=np.random.default_rng(random_seed)
    )


def createNextStep(
    progression: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
    infectiousContacts: Dict[NodeName, Dict[Age, float]],
    currState: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
    infectionProb: float,
    stochastic: bool,
    random_state: Optional[np.random.Generator]
) -> Dict[NodeName, Dict[Tuple[Age, Compartment], float]]:
    """Update the current state of each regions population by allowing infected individuals
    to progress to the next infection stage and infecting susceptible individuals. The state is not
    modified in this function, rather the updated details are returned.

    :param progression: The number of individuals that have progressed into each exposed stage, stratified by
                        region and age.
    :param infectiousContacts: The number of contacts per region per age.
    :param currState: The current state for every region.
    :param infectionProb: the expected rate at which contacts will transmit the diseases
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    :return: The new state of the regions.
    """

    assert progression.keys() == infectiousContacts.keys() == currState.keys(), "missing regions"

    nextStep: Dict[NodeName, Dict[Tuple[Age, Compartment], float]] = {}
    for name, node in currState.items():
        new_node = nextStep.setdefault(name, {})
        for key, value in node.items():
            # We need to keep the susceptibles in order to infect them
            if key[1] == SUSCEPTIBLE_STATE:
                new_node[key] = value
            else:
                new_node[key] = 0.0

    for name, node in progression.items():
        for key, value in node.items():
            # Note that the progression is responsible for populating every other state
            assert key[1] != SUSCEPTIBLE_STATE, "Susceptibles can't be part of progression states"
            nextStep[name][key] = value

    for name, node in infectiousContacts.items():
        for age, infectiousContacts in node.items():
            susceptible = nextStep[name][(age, SUSCEPTIBLE_STATE)]
            exposed = _calculateExposed(susceptible, infectiousContacts, infectionProb, stochastic, random_state)
            expose(age, exposed, nextStep[name])

    return nextStep


def _calculateExposed(
    susceptible: float,
    contacts: float,
    infectionProb: float,
    stochastic: bool,
    random_state: Optional[np.random.Generator]
):
    """From the number of contacts (between infectious and susceptible) people, compute
    the number of actual infectious contacts. Two modes:
    1) Deterministic
    Adjusts the contacts (see explanation below), then multiply by infection probability
    2) Stochastic
    Adjusts the contacts (see explanation below), then sample from a binomial distribution
    with n=adjustedContacts, p=infectionProb.

    When using k contacts from a susceptible population of size n, we sample
    these k people WITH replacement, as in several infections can target the same person.
    This will decrease the exposed number for small values of k, n. This is not sampled
    stochastically as it seems no such distribution exists.

    E[Number of different people chosen when picking k in a population of size n, with replacement]
    = sum_{i=1,...,n} P(person i is chosen at least once)
    = sum_{i=1,...,n} (1 - P(person i is never chosen in k samples))
    = sum_{i=1,...,n} (1 - P(person i is not chosen once)^k)
    = sum_{i=1,...,n} (1 - (1 - P(person i is chosen once))^k)
    = sum_{i=1,...,n} (1 - (1 - P(person i is chosen once))^k)
    = n * (1 - (1 - 1 / n)^k)

    :param susceptible: number (float) of susceptible individuals
    :param contacts: number (float) of exposed individuals
    :param infectionProb: probability (float) that a given contact will result in infection
    :param stochastic: Whether to run the model in a stochastic or deterministic mode
    :param random_state: Random number generator used for the model
    """
    if susceptible > 1.:
        probaNeverChosen = (1 - (1 / susceptible)) ** contacts
        adjustedContacts = susceptible * (1 - probaNeverChosen)
    else:
        # For susceptible < 1., the formula (which theoretically works
        # only for integers) breaks down and returns nan. In that case
        # several choices are acceptable, we assume the fractional person will
        # be chosen with 100% probability, taking the minimum to ensure
        # exposed < susceptible is always true.
        adjustedContacts = contacts

    if stochastic:
        # AdjustedContacts can be non integer because of the adjustment, round it
        infectiousContacts = stats.binom.rvs(int(round(adjustedContacts)), infectionProb, random_state=random_state)
    else:
        infectiousContacts = adjustedContacts * infectionProb

    return min(infectiousContacts, susceptible)


def getSusceptibles(age: str, currentInternalStateDict: Dict[Tuple[Age, Compartment], float]) -> float:
    """Calculate the total number of individuals in a susceptible state in an age range within a
    region.

    :param age: The age (range)
    :param currentInternalStateDict: The disease status of the population in a region stratified
                                     by age.
    :return: The number of individuals in a susceptible state and age range.
    """
    return currentInternalStateDict[(age, SUSCEPTIBLE_STATE)]


def expose(age: Age, exposed: float, region: Dict[Tuple[Age, Compartment], float]):
    """Update the region in place, moving people from susceptible to exposed.

    :param age: age group that will be exposed.
    :param exposed: The number of exposed individuals.
    :param region: A region, with all the (age, state) tuples.
    """
    assert region[(age, SUSCEPTIBLE_STATE)] >= exposed, f"S:{region[(age, SUSCEPTIBLE_STATE)]} < E:{exposed}"

    region[(age, EXPOSED_STATE)] += exposed
    region[(age, SUSCEPTIBLE_STATE)] -= exposed


def randomlyInfectRegions(
        network: NetworkOfPopulation,
        regions: int,
        age_groups: List[Age],
        infected: float,
) -> Dict[NodeName, Dict[Age, float]]:
    """Randomly infect regions to initialize the random simulation

    :param network: object representing the network of populations
    :param regions: The number of regions to expose.
    :param age_groups: Age groups to infect
    :param infected: People to infect
    :return: Structure of initially infected regions with number
    """
    infections = {}
    for regionID in random.choices(list(network.graph.nodes()), k=regions):
        infections[regionID] = {}
        for age in age_groups:
            infections[regionID][age] = infected

    return infections
