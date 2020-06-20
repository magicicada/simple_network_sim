import logging
import math
import random
from typing import Dict, Tuple, NamedTuple, List, Set

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import ListedColormap

from simple_network_sim import loaders

logger = logging.getLogger(__name__)

# Type aliases used to make the types for the functions below easier to read
Age = str
Compartment = str
NodeName = str
Time = int


class NetworkOfPopulation(NamedTuple):
    """
    This type has all the internal data used by this model
    """
    progression: Dict[Age, Dict[Compartment, Dict[Compartment, float]]]
    states: Dict[Time, Dict[NodeName, Dict[Tuple[Age, Compartment], float]]]
    graph: nx.DiGraph
    mixingMatrix: loaders.MixingMatrix
    movementMultipliers: Dict[Time, loaders.Multiplier]
    infectiousStates: List[Compartment]
    infectionProb: Dict[Time, float]


# CurrentlyInUse
def countInfectiousAgeStructured(
    nodes: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
    infectiousStates: List[Compartment]
) -> float:
    """Count the number of infectious individuals in all nodes at some time.

    :param nodes: A dictionary of nodes.
    :param time: The time within the simulation.
    :type time: int
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: The total number of infectious individuals in all regions.
    :rtype: float
    """
    total = 0
    for node in nodes.values():
        for age in getAges(node):
            total += getInfectious(age, node, infectiousStates)
    return total


# NotCurrentlyInUse
# this takes a dictionary of states at times at nodes, and returns a string
# reporting the number of people in each state at each node at each time.
# aggregates by age 
def basicReportingFunction(dictOfStates):
    reportString = ""
    dictOfStringsByNodeAndState = {}
#     Assumption: all nodes exist at time 0
    for node in dictOfStates[0]:
        dictOfStringsByNodeAndState[node] = {}
        for (age, state) in dictOfStates[0][node]:
            dictOfStringsByNodeAndState[node][state] = []
    for time in dictOfStates:
        for node in dictOfStates[time]:
            numByState = {}
            for (age, state) in dictOfStates[time][node]:
                if state not in numByState:
                    numByState[state] = 0
                numByState[state] = numByState[state] + dictOfStates[time][node][(age, state)]
            for state in numByState:
                dictOfStringsByNodeAndState[node][state].append(numByState[state])
           
    logger.debug(dictOfStringsByNodeAndState)
    
    for node in dictOfStringsByNodeAndState:
        for state in dictOfStringsByNodeAndState[node]:
            localList = dictOfStringsByNodeAndState[node][state]
            localString = ""
            for elem in localList:
                localString = localString + "," + str(elem)
            reportString = reportString+"\n" + str(node) + "," + str(state) + localString
    return reportString


# CurrentlyInUse
def basicSimulationInternalAgeStructure(network, timeHorizon):
    """Run the simulation of a disease progressing through a network of regions.

    :param network: This is a NetworkOfPopulation instance which will have the states field modified
    by this function.
    :type network: A NetworkOfPopulation object.
    :param timeHorizon: How many times to run the simulation. Each new time means a new entry to the
    network.states dict
    :type timeHorizon: int
    :return: A time series of the size of the infectious population.
    :rtype: A list with the number of infectious people at each given time
    """
    timeSeriesInfection = []
    multipliers = network.movementMultipliers.get(0, loaders.Multiplier(contact=1.0, movement=1.0))
    infectionProb = network.infectionProb[0]  # no default value, time zero must exist
    for time in range(timeHorizon):
        # we are building the interactions for time + 1, so that's the multiplier value we need to use
        multipliers = network.movementMultipliers.get(time + 1, multipliers)
        infectionProb = network.infectionProb.get(time + 1, infectionProb)

        progression = getInternalProgressionAllNodes(network.states[time], network.progression)

        internalContacts = getInternalInfectiousContacts(
            network.states[time],
            network.mixingMatrix,
            multipliers.contact,
            network.infectiousStates,
        )
        externalContacts = getExternalInfectiousContacts(
            network.graph,
            network.states[time],
            multipliers.movement,
            network.infectiousStates,
        )
        contacts = mergeContacts(internalContacts, externalContacts)

        network.states[time + 1] = createNextStep(progression, contacts, network.states[time], infectionProb)

        timeSeriesInfection.append(countInfectiousAgeStructured(network.states[time], network.infectiousStates))

    return timeSeriesInfection


# CurrentlyInUse
def totalIndividuals(nodeState):
    """This function takes a node (region) and counts individuals from every age and state.

    :param nodeState: The disease status of a node (or region) stratified by age.
    :type nodeState: A dictionary with a tuple of (age, state) as keys and the number of individuals
    in that state as values.
    :return: The total number of individuals.
    :rtype: float
    """
    return sum(nodeState.values())


def getAges(node: Dict[Tuple[Age, Compartment], float]) -> Set[Age]:
    """Get the set of ages from the node.

    :param node: The disease states of the population stratified by age.
    :type node: A dictionary with a tuple of (age, state) as keys and the number of individuals
    in that state as values.
    :return: The unique collection of ages.
    :rtype: Set[str]
    """
    ages = set()
    for (age, state) in node:
        ages.add(age)
    return ages


# CurrentlyInUse
def getTotalInAge(nodeState, ageTest):
    """Get the size of the population within an age group.

    :param nodeState: The disease states of the population stratified by age.
    :type nodeState: A dictionary with a tuple of (age, state) as keys and the number of individuals
    in that state as values.
    :param ageTest: The age range of the population.
    :type ageTest: str, e.g. '70+'
    :return: The population size within the age range.
    :rtype: int
    """
    total = 0
    for (age, _), value in nodeState.items():
        if age == ageTest:
            total += value
    return total


# CurrentlyInUse
def getTotalInfectious(node: Dict[Tuple[Age, Compartment], float], infectiousStates: List[Compartment]) -> float:
    """Get the total number of infectious individuals regardless of age in the node.

    :param node: The disease status of the population stratified by age.
    :type node: A dictionary with a tuple of (age, state) as keys and the number of individuals
    in that state as values.
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: The total number of infectious individuals.
    :rtype: float
    """
    total = 0.0
    for (_, compartment), value in node.items():
        if compartment in infectiousStates:
            total += value
    return total


# CurrentlyInUse
def getTotalSuscept(nodeState):
    """Get the total number of susceptible individuals regardless of age in the node

    :param nodeState: The disease status of the population stratified by age.
    :type nodeState: A dictionary with a tuple of (age, state) as keys and the number of individuals
    in that state as values.
    :return: The total number of susceptible individuals.
    :rtype: float
    """
    totalSusHere = 0.0
    for age in getAges(nodeState):
        totalSusHere += getSusceptibles(age, nodeState)
    return totalSusHere


# CurrentlyInUse
def distributeContactsOverAges(nodeState, newInfections):
    """Distribute the number of new infection cases across a region. This function distributes
    newInfections among the susceptibles according to the relative size of each age group. Note:
    fractional people will come out of this.

    :param nodeState: The disease status of the population stratified by age.
    :type nodeState: A dictionary with a tuple of (age, state) as keys and the number of individuals
    in that state as values.
    :param newInfections: The number of new infections to be distributed across age ranges.
    :type newInfections: int
    :return: The number of new infections in each age group.
    :rtype: A dictionary of ages (keys) and the number of new infections (values)
    """
    ageToSus = {}
    newInfectionsByAge = {}
    totalSus = 0.0
    for age in getAges(nodeState):
        sus = getSusceptibles(age, nodeState)
        ageToSus[age] = sus
        totalSus += sus
    if totalSus < newInfections:
        logger.error("totalSus < incoming infectious (%s < %s) - adjusting to totalSus", totalSus, newInfections)
        newInfections = totalSus
    for age, sus in ageToSus.items():
        if totalSus > 0:
            newInfectionsByAge[age] = (sus / totalSus) * newInfections
        else:
            newInfectionsByAge[age] = 0.0
    return newInfectionsByAge


def getIncomingInfectiousContactsByNode(graph, currentState, movementMultiplier, infectiousStates):
    """Determine the number of new infections at each node of a graph based on incoming people
    from neighbouring nodes.

    :param graph: A graph with each region as a node and the weights corresponding to the movements
    between regions. Edges must contain weight and delta_adjustment attributes (assumed 1.0)
    :type graph: networkx.Digraph
    :param currentState: The current state for every region.
    :type currentState: A dictionary with the region as a key and the value is a dictionary of
    states in the format {(age, state): number of individuals in this state}.
    :param movementMultiplier: a multiplier applied to each edge (movement) in the network.
    :type movementMultiplier: float
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: the number of new infections in each region.
    :rtype: A dictionary with the region as key and the number of new infections as the value.
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
                    contactsByNode[receivingVertex] += weight * fractionGivingInfected * fractionReceivingSus

    return contactsByNode


def getWeight(graph, orig, dest, multiplier):
    """Get the weight of the edge from orig to dest in the graph. This weight is expected to be
    proportional to the movement between nodes. If the edge doesn't have a weight, 1.0 is assumed
    and the returned weight is adjusted by the multiplier and any delta_adjustment on the edge.

    :param graph: A graph with each region as a node and the weights corresponding to the commutes
    between regions. Edges must contain weight and delta_adjustment attributes (assumed 1.0)
    :type graph: networkx.DiGraph
    :param orig: The vertex people are coming from.
    :type orig: str
    :param dest: The vertex people are going to.
    :type dest: str
    :param multiplier: Value that will dampen or heighten movements between nodes.
    :type multiplier" float
    :return: The final weight value
    :rtype: float
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
    # enable it fully (factor = 1.0). If the movement multipler doesn't make any changes to the node's movements (ie.
    # multiplier = 1.0), then the delta_adjustment will have no effect.
    return weight - (delta * delta_adjustment)


# CurrentlyInUse
# To bring this in line with the within-node infection updates (and fix a few bugs), I'm going to rework
# it so that we calculate an *expected number* of infectious contacts more directly. Then we'll distribute and
# overlap them using the same infrastructure code that we'll use for the internal version, when we add that
# Reminder: I expect the weighted edges to be the number of *expected infectious* contacts (if the giver is infectious)
#  We may need to multiply movement numbers by a probability of infection to achieve this.   
def getExternalInfectiousContacts(graph, nodes, movementMultiplier, infectiousStates):
    """Calculate the number of new infections in each region. The infections are distributed
    proportionally to the number of susceptibles in the destination node and infected in the origin
    node. The infections are distributed to each age group according to the number of susceptible
    people in them.

    :param graph: A graph with each region as a node and the weights corresponding to the movements
    between regions. Edges must contain weight and delta_adjustment attributes (assumed 1.0)
    :type graph: networkx.Digraph
    :param dictOfStates: A time series of the disease status in each region stratified by age.
    :type dictOfStates: A dictionary with time as keys and whose values are another dictionary with
    the region as a key and the disease state as values. The states are a dictionary with a tuple
    of (age, state) as keys and the number of individuals in that state as values.
    :param currentTime: The time (at which we calculate the infections from outside the region).
    :type currentTime: int
    :param movementMultiplier: A multiplier applied to each edge (movement between nodes) in the
    network.
    :type movementMultiplier: float
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: The number of new infections in each region stratified by age.
    :rtype: A dictionary with the region as a key and a dictionary of {age: number of new
    infections} as values.
    """
    infectionsByNode = {}

    incomingContacts = getIncomingInfectiousContactsByNode(graph, nodes, movementMultiplier, infectiousStates)

    for name, vertex in incomingContacts.items():
        infectionsByNode[name] = distributeContactsOverAges(nodes[name], vertex)

    return infectionsByNode


# CurrentlyInUse
#  (JE, 10 May 2020) I'm realigning this to be more using with a POLYMOD-style matrix (including the within-lockdown
#  COMIX matrix) I expect the matrix entry at [age1][age2] to be the expected number of contacts in a day between age1
#  and age2, that would infect if only one end of the contact were infectious.
#  That is, if a usual POLYMOD entry tells us that each individual of age1 is expected to have 1.2 contacts in category
#  age2, and the probability of each of these being infectious is 0.25, then I would expect the matrix going into this
# function as ageMixingInfectionMatrix to have 0.3 in the entry [age1][age2]
def getInternalInfectiousContactsInNode(
    currentInternalStateDict,
    mixingMatrix,
    contactsMultiplier,
    infectiousStates,
):
    """Calculate the new infections due to mixing within the region and stratify them by age.

    :param currentInternalStateDict: The disease status of the population stratified by age.
    :type currentInternalStateDict: A dict with a tuple of (age, state) as keys and the
    number of individuals in that state as values.
    :param mixingMatrix: Stores expected numbers of interactions between people of
    different ages.
    :type mixingMatrix: A dict with age range object as a key and Mixing Ratio as
    a value.
    :param contactsMultiplier: Multiplier applied to the number of infectious contacts.
    :type contactsMultiplier: float
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: The number of new infections stratified by age.
    :rtype: A dictionary of {age: number of new infections}
    """
    infectiousContacts: Dict[Age, float] = {}
    for age in mixingMatrix:
        infectiousContacts[age] = 0

        numSuscept = getSusceptibles(age, currentInternalStateDict)
        if numSuscept > 0:
            contacts = 0
            for ageInf in mixingMatrix[age]:
                contacts += getInfectious(ageInf, currentInternalStateDict, infectiousStates) * mixingMatrix[ageInf][age]

            totalInAge = getTotalInAge(currentInternalStateDict, age)
            if totalInAge > 0.0:
                infectiousContacts[age] = contacts * (numSuscept / totalInAge) * contactsMultiplier
            else:
                infectiousContacts[age] = 0.0

    return infectiousContacts


# CurrentlyInUse        
def getInternalInfectiousContacts(nodes, mixingMatrix, contactsMultiplier, infectiousStates):
    """Calculate the new infections and stratify them by region and age.

    :param dictOfStates: A time series of the disease status in each region stratified by age.
    :type dictOfStates: A dictionary with time as keys and whose values are another dictionary with
    the region as a key and the disease state as values. The states are a dictionary with a tuple
    of (age, state) as keys and the number of individuals in that state as values.
    :param mixingMatrix: Stores expected numbers of interactions between people of different ages.
    :type mixingMatrix: Dictionary with age range object as a key and Mixing Ratio as
    a value, essentially a dict with age range object as a key and Mixing Ratio as
    a value.
    :param time: The time
    :type time: int
    :param contactsMultiplier: Multiplier applied to the number of infectious contacts.
    :type contactsMultiplier: float
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: The number of exposed in each region stratified by age.
    :rtype: A dictionary containing the region as a key and a dictionary of {age: number of
    exposed} as a value.
    """

    contacts: Dict[NodeName, Dict[Age, float]] = {}

    for name, node in nodes.items():
        contacts[name] = getInternalInfectiousContactsInNode(node, mixingMatrix, contactsMultiplier, infectiousStates)

    return contacts


# CurrentlyInUse
def internalStateDiseaseUpdate(currentInternalStateDict, diseaseProgressionProbs):
    """Returns the status of exposed individuals, moving them into the next disease state with a
    probability defined in the given progression matrix.

    :param currentInternalStateDict: The disease status of the population stratified by age.
    :type currentInternalStateDict: A dictionary with keys (age, state) and the number of
    individuals in that node in that state as values.
    :param diseaseProgressionProbs: A matrix with the probabilities if progressing from one state
    to the next.
    :type diseaseProgressionProbs: Nested dictionary with the format.
    {age : {disease state 1: {disease state 2: probability of progressing from state 1 to state 2}}}
    :return: the numbers in each exposed state stratified by age
    :rtype: A dictionary of { (age: state) : number in this state}. Note this only contains
    exposed states.
    """
    newStates = {}
    for (age, state), people in currentInternalStateDict.items():
        outTransitions = diseaseProgressionProbs[age].get(state, {})
        for nextState in outTransitions:
            newStates.setdefault((age, nextState), 0.0)
            newStates[(age, nextState)] += outTransitions[nextState] * people
    return newStates


# CurrentlyInUse
def getInternalProgressionAllNodes(currStates, diseaseProgressionProbs):
    """Given the size of the population in each exposed state, calculate the numbers that progress
    the next disease state based on the progression matrix.

    :param currStates: The current state for every region is not modified.
    :type currStates: A dictionary with the region as a key and the value is a dictionary in the
    format {(age, state): number of individuals in this state}.
    :param diseaseProgressionProbs: A matrix with the probabilities if progressing from one state
    to the next.
    :type diseaseProgressionProbs: Nested dictionary with the format
    {age : {disease state 1: {disease state 2: probability of progressing from state 1 to state 2}}}
    :return: The number of individuals that have progressed into each exposed stage, stratified by
    region and age.
    :rtype: A dictionary with region as keys and a dictionary of { (age: state) : number in this
    state} as values, note these states do not include susceptible. Note this only contains
    exposed states.
    """
    progression = {}
    for regionID, currRegion in currStates.items():
        progression[regionID] = internalStateDiseaseUpdate(currRegion, diseaseProgressionProbs)
    return progression


def mergeContacts(*args):
    """From a list of exposed cases stratified by age and region, merge them into a single
    collection

    :param args: A variable list of regional exposure numbers by age.
    :type args: A variable list of dictionaries that have a region as a key and the value is a
    dictionary of {age:number of exposed}.
    :return: A dictionary containing the merged number of exposed.
    :rtype: A dictionary with a region as a key and the whose value is a dictionary of age:number of
    exposed.
    """
    exposedTotal = {}
    for infectionsDict in args:
        for regionID, region in infectionsDict.items():
            exposedRegionTotal = exposedTotal.setdefault(regionID, {})
            for age, exposed in region.items():
                exposedRegionTotal[age] = exposedRegionTotal.setdefault(age, 0.0) + exposed
    return exposedTotal


def exposeRegions(infections, states):
    """Update the state of a region, adding new infections. This function modifies the state
    parameter.

    :param infections: The number of infections per region per age
    :type infections: A dictionary with region as keys and a dictionary in the format
    {age: the number of infections} as values.
    :param states: The current state for every region is not modified
    :type states: A dictionary with the region as a key and the value is a dictionary in the format
    {(age, state): number of individuals in this state}.
    """
    for nodeName, node in infections.items():
        for age, value in node.items():
            expose(age, value, states[nodeName])


def modelStatesToPandas(timeseries: Dict[int, Dict[str, Dict[Tuple[str, str], float]]]) -> pd.DataFrame:
    """Takes an instance of NetworkOfPopulations.states and transforms it into a pandas DataFrame.

    :param timeseries: dict object with the state over the times
    :type timeseries: A dictionary in the format
    {time : {region: { (age, state): population size}}}
    :return: a pandas dataframe in tabular format with the following columns:
             - time
             - node
             - age
             - state
             - total
    """
    rows = []
    for time, nodes in timeseries.items():
        for nodeID, node in nodes.items():
            for (age, state), value in node.items():
                rows.append({"time": time, "node": nodeID, "age": age, "state": state, "total": value})
    return pd.DataFrame(rows)


def plotStates(df, nodes=None, states=None, ncol=3, sharey=False, figsize=None, cmap=None):
    """
    Plots a grid of plots, one plot per node, filtered by disease progression states (each states will be a line). The
    graphs are all Number of People x Time

    :param df: pandas DataFrame with node, time, state and total columns
    :param nodes: creates one plot per nodes listed (None means all nodes)
    :param df: pandas DataFrame with nodes, time, state and total columns
    :type df: pandas DataFrame
    :param nodes: creates one plot per node listed (None means all nodes)
    :type nodes: list (of region names).
    :param states: plots one curve per state listed (None means all states)
    :type states: list (of disease states).
    :param ncol: number of columns (the number of rows will be calculated to fit all graphs)
    :type ncol: int
    :param sharey: set to true if all plots should have the same y-axis
    :type sharey: bool
    :param figsize: select the size of each individual plot
    :type figsize:
    :param cmap: color map to use
    :type cmap:
    :return: returns a matplotlib figure
    :rtype: matplotlib figure
    """
    if nodes is None:
        nodes = df.node.unique().tolist()
    if states is None:
        states = df.state.unique().tolist()
    if cmap is None:
        cmap = ListedColormap(["#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#E69F00"])
    nrow = math.ceil(len(nodes) / ncol)
    if figsize is None:
        if nrow >= 3:
            figsize = (20, 20)
        elif nrow == 2:
            figsize = (20, 10)
        else:
            figsize = (20, 5)

    if not nodes:
        raise ValueError("nodes cannot be an empty list")
    if not states:
        raise ValueError("states cannot be an empty list")

    # pre filter by states
    df = df[df.state.isin(states)]

    fig, axes = plt.subplots(nrow, ncol, squeeze=False, constrained_layout=True, sharey=sharey, figsize=figsize)

    count = 0
    ax = None
    for i in range(nrow):
        for j in range(ncol):
            if count < len(nodes):
                node = nodes[count]
                count += 1
                grouped = df[df.node == node].groupby(["time", "state"]).sum()
                indexed = grouped.reset_index().pivot(index="time", columns="state", values="total")

                ax = axes[i, j]
                indexed.plot(ax=ax, legend=False, title=node, cmap=cmap)
                ax.set_ylabel("Number of People")
                ax.set_xlabel("Time")

    assert ax is not None, "ax was never assigned"
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    return fig


def getInfectious(age, currentInternalStateDict, infectiousStates):
    """Calculate the total number of individuals in infectious states in an age range.

    :param age: The age (range)
    :type age: str, e.g. '[17,70)'
    :param currentInternalStateDict: The disease status of the population stratified by age.
    :type currentInternalStateDict: A dictionary containing a tuple of age range (as a string)
    and a disease state as a key and the number of individuals in the (age,state) as a value.
    :param infectiousStates: States that are considered infectious
    :type infectiousStates: list of strings
    :return: The number of individuals in an infectious state and age range.
    :rtype: float
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
    movement_multipliers_table: pd.DataFrame = None,
) -> NetworkOfPopulation:
    """Create the network of the population, loading data from files.

    :param compartment_transition_table: pd.Dataframe specifying the transition rates between infected compartments.
    :param population_table: pd.Dataframe with the population size in each region by gender and age.
    :param commutes_table: pd.Dataframe with the movements between regions.
    :param mixing_matrix_table: pd.Dataframe with the age infection matrix.
    :param movement_multipliers_table: pd.Dataframe with the movement multipliers. This may be None, in
    which case no multipliers are applied to the movements.
    :param infectious_states: States that are considered infectious
    :param infection_prob: Probability that a given contact will result in an infection
    :return: The constructed network
    """
    infection_prob = loaders.readInfectionProbability(infection_prob)

    infectious_states = loaders.readInfectiousStates(infectious_states)
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
        f"mismatched infectious states and states {infectious_states} {set(progression.keys())}"

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
    assert set(graph.nodes()) == set(population.keys()), "regions mismatch between graph and population"

    state0: Dict[str, Dict[Tuple[str, str], float]] = {}
    for node in list(graph.nodes()):
        region = state0.setdefault(node, {})
        for age, compartments in progression.items():
            region[(age, SUSCEPTIBLE_STATE)] = population[node][age]
            for compartment in compartments:
                region[(age, compartment)] = 0

    return NetworkOfPopulation(
        progression=progression,
        graph=graph,
        states={0: state0},
        mixingMatrix=mixingMatrix,
        movementMultipliers=movementMultipliers,
        infectiousStates=infectious_states,
        infectionProb=infection_prob,
    )


def createNextStep(
    progression: Dict[Age, Dict[Compartment, Dict[Compartment, float]]],
    infectiousContacts: Dict[NodeName, Dict[Age, float]],
    currState: Dict[NodeName, Dict[Tuple[Age, Compartment], float]],
    infectionProb: float,
) -> Dict[NodeName, Dict[Tuple[Age, Compartment], float]]:
    """Update the current state of each regions population by allowing infected individuals
    to progress to the next infection stage and infecting susceptible individuals. The state is not
    modified in this function, rather the updated details are returned.

    :param progression: The number of individuals that have progressed into each exposed stage, stratified by
    region and age.
    :param infectiousContacts: The number of contacts per region per age.
    :param currState: The current state for every region.
    :param infectionProb: the expected rate at which contacts will transmit the diseases
    :return: The new state of the regions.
    """

    assert progression.keys() == infectiousContacts.keys() == currState.keys(), "missing regions"

    nextStep = {}
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
            exposed = _calculateExposed(susceptible, infectiousContacts, infectionProb)
            expose(age, exposed, nextStep[name])

    return nextStep


def _calculateExposed(susceptible: float, contacts: float, infectionProb: float):
    """
    :param susceptible: number (float) of susceptible individuals
    :param contacts: number (float) of exposed individuals
    :param infectionProb: probability (float) that a given contact will result in infection

    When modelling infections in k people from a susceptible population of size n, we sample
    these k people WITH replacement, as in several infections can target the same person.
    This will decrease the exposed number for small values of k, n.

    E[Number of different people chosen when picking k in a population of size n, with replacement]
    = sum_{i=1,...,n} P(person i is chosen at least once)
    = sum_{i=1,...,n} (1 - P(person i is never chosen in k samples))
    = sum_{i=1,...,n} (1 - P(person i is not chosen once)^k)
    = sum_{i=1,...,n} (1 - (1 - P(person i is chosen once))^k)
    = sum_{i=1,...,n} (1 - (1 - P(person i is chosen once))^k)
    = n * (1 - (1 - 1 / n)^k)
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

    return min(adjustedContacts * infectionProb, susceptible)


def getSusceptibles(age, currentInternalStateDict):
    """Calculate the total number of individuals in a susceptible state in an age range within a
    region.

    :param age: The age (range)
    :type age: str, e.g. '[17,70)'
    :param currentInternalStateDict: The disease status of the population in a region stratified
    by age.
    :type currentInternalStateDict: A dictionary containing a tuple of age range (as a string)
    and a disease state as a key and the number of individuals in the (age,state) as a value.
    :return: The number of individuals in a susceptible state and age range.
    :rtype: float
    """
    return currentInternalStateDict[(age, SUSCEPTIBLE_STATE)]


def expose(age, exposed, region):
    """Update the region in place, moving people from susceptible to exposed.

    :param age: age group that will be exposed.
    :type age: str, e.g. '[17,70)'.
    :param exposed: The number of exposed individuals.
    :type exposed: float.
    :param region: A region, with all the (age, state) tuples.
    :type region: a dictionary containing a tuple of age range (as a string) as a key and the
    number of individuals in the (age,state) as a value.
    """
    assert region[(age, SUSCEPTIBLE_STATE)] >= exposed, f"S:{region[(age, SUSCEPTIBLE_STATE)]} < E:{exposed}"

    region[(age, EXPOSED_STATE)] += exposed
    region[(age, SUSCEPTIBLE_STATE)] -= exposed


def randomlyInfectRegions(network, regions, age_groups, infected):
    """Randomly infect regions to initialize the random simulation

    :param network: object representing the network of populations
    :type network: A NetworkOfPopulation object
    :param regions: The number of regions to expose.
    :type regions: int
    :param age_groups: Age groups to infect
    :type age_groups: list
    :param infected: People to infect
    :type infected: int
    :return: Structure of initially infected regions with number
    :rtype: dict
    """
    infections = {}
    for regionID in random.choices(list(network.graph.nodes()), k=regions):
        infections[regionID] = {}
        for age in age_groups:
            infections[regionID][age] = infected

    return infections


# NotCurrentlyInUse
def nodeUpdate(graph, dictOfStates, time, headString):
    print('\n\n===== BEGIN update 1 at time ' + str(time) + '=========' + headString)
    for node in list(graph.nodes()):
        print('Node ' + str(node) + " E-A-I at mature " + str(dictOfStates[time][node][('m', 'E')]) + " " +
              str(dictOfStates[time][node][('m', 'A')]) + " " + str(dictOfStates[time][node][('m', 'I')]))
    print('===== END update 1 at time ' + str(time) + '=========')
