import copy
import logging
import math
from typing import Dict, Tuple, NamedTuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.colors import ListedColormap

from simple_network_sim import loaders

logger = logging.getLogger(__name__)

# CurrentlyInUse
def countInfectiousAgeStructured(dictOfStates, time):
    total = 0
    for node in dictOfStates[time]:
        for age in getAges(dictOfStates[time][node]):
            total += getInfectious(age, dictOfStates[time][node])
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
    """
    :param network: this is a NetworkOfPopulation instance which will have the states field modified by this function
    :param timeHorizon: how many times to run the simulation. Each new time means a new entry to the network.states dict
    :return: a list with the number of infectious people at each given time
    """
    timeSeriesInfection = []
    movementMultiplier = network.movementMultipliers.get(0, 1.0)
    for time in range(timeHorizon):
        # we are building the interactions for time + 1, so that's the multiplier value we need to use
        movementMultiplier = network.movementMultipliers.get(time + 1, movementMultiplier)

        progression = getInternalProgressionAllNodes(network.states[time], network.progression)

        internalInfections = getInternalInfection(network.states, network.infectionMatrix, time)
        externalInfections = getExternalInfections(network.graph, network.states, time, movementMultiplier)
        exposed = mergeExposed(internalInfections, externalInfections)

        network.states[time + 1] = createNextStep(progression, exposed, network.states[time])

        timeSeriesInfection.append(countInfectiousAgeStructured(network.states, time))

    return timeSeriesInfection


# CurrentlyInUse
def totalIndividuals(nodeState):
    return sum(nodeState.values())


def getAges(node):
    ages = set()
    for (age, state) in node:
        ages.add(age)
    return ages


# CurrentlyInUse
def getTotalInAge(nodeState, ageTest):
    total= 0
    for (age, state) in nodeState:
            if age == ageTest:
                total = total + nodeState[(age, state)]
    return total


# CurrentlyInUse
def getTotalInfectious(nodeState):
    totalInfectedHere = 0.0
    for age in getAges(nodeState):
        totalInfectedHere += getInfectious(age, nodeState)
    return totalInfectedHere


# CurrentlyInUse
def getTotalSuscept(nodeState):
    totalSusHere = 0.0
    for age in getAges(nodeState):
        totalSusHere += getSusceptibles(age, nodeState)
    return totalSusHere


# CurrentlyInUse
# fractional people will come out of this
# right now this infects uniformly across age class by number of susceptibles in age class 
def distributeInfections(nodeState, newInfections):
    ageToSus = {}
    newInfectionsByAge = {}
    for age in getAges(nodeState):
        ageToSus[age] = getSusceptibles(age, nodeState)
    totalSus = getTotalSuscept(nodeState)
    if totalSus<newInfections:
        logger.error('Too many infections to distribute amongst age classes - adjusting num infections')
        newInfections = totalSus
    for age in ageToSus:
        if totalSus > 0:
            newInfectionsByAge[age] = (float(ageToSus[age])/float(totalSus))*newInfections
        else:
            newInfectionsByAge[age] = 0
    return newInfectionsByAge


def doIncomingInfectionsByNode(graph, currentState, movementMultiplier):
    totalIncomingInfectionsByNode = {}
    for receivingVertex in currentState:
        totalSusceptHere = getTotalSuscept(currentState[receivingVertex])
        totalIncomingInfectionsByNode[receivingVertex] = 0
        if totalSusceptHere > 0:
            neighbours = list(graph.predecessors(receivingVertex))
            for givingVertex in neighbours:
                if givingVertex == receivingVertex:
                    continue
                totalInfectedGiving = getTotalInfectious(currentState[givingVertex])
                if totalInfectedGiving > 0:
                    weight = getWeight(graph, givingVertex, receivingVertex, movementMultiplier)

                    fractionGivingInfected = totalInfectedGiving / totalIndividuals(currentState[givingVertex])
                    fractionReceivingSus = totalSusceptHere / totalIndividuals(currentState[receivingVertex])
                    totalIncomingInfectionsByNode[receivingVertex] += weight * fractionGivingInfected * fractionReceivingSus

    return totalIncomingInfectionsByNode


def getWeight(graph, orig, dest, multiplier):
    """
    :param graph: the graph with the commutes, weight and multiplier
    :param orig: vertex people are coming from
    :param dest: vertex people are going to
    :param multiplier: value that will dampen or heighten movements between nodes
    :return: a float with the final weight value
    """
    edge  = graph[orig][dest]
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
def getExternalInfections(graph, dictOfStates, currentTime, movementMultiplier):
    infectionsByNode = {}

    totalIncomingInfectionsByNode = doIncomingInfectionsByNode(graph, dictOfStates[currentTime], movementMultiplier)

    # This might over-infect - we will need to adapt for multiple infections on a single individual if we have high infection threat.  TODO raise an issue
    for vertex in totalIncomingInfectionsByNode:
        totalDelta = totalIncomingInfectionsByNode[vertex]
        infectionsByNode[vertex] = distributeInfections(dictOfStates[currentTime][vertex], totalDelta)

    return infectionsByNode


# CurrentlyInUse
#  (JE, 10 May 2020) I'm realigning this to be more using with a POLYMOD-style matrix (inlcuding the within-lockdown COMIX matrix)
# I expect the matrix entry at [age1][age2] to be the expected number of contacts in a day between age1 and age2
#  *that would infect if only one end of the contact were infectious*
#  that is, if a usual POLYMOD entry tells us that each individual of age1 is expected to have 1.2 contacts in category age2,
#  and the probability of each of these being infectious is 0.25, then I would expect the matrix going into this
# function as  ageMixingInfectionMatrix to have 0.3 in the entry [age1][age2]
def doInternalInfectionProcess(currentInternalStateDict, ageMixingInfectionMatrix):
    newInfectedsByAge = {}
    for age in ageMixingInfectionMatrix:
        newInfectedsByAge[age] = 0

        numSuscept = getSusceptibles(age, currentInternalStateDict)
        if numSuscept>0:
            numInfectiousContactsFromAges = {}
            totalNewInfectionContacts = 0
            for ageInf in ageMixingInfectionMatrix[age]:
                totalInfectious = getInfectious(ageInf, currentInternalStateDict)
                # TODO: Make sure this is not implemented in a slow way anymore after https://github.com/ScottishCovidResponse/SCRCIssueTracking/issues/273
                numInfectiousContactsFromAges[ageInf] = totalInfectious*ageMixingInfectionMatrix[ageInf][age]
                totalNewInfectionContacts = totalNewInfectionContacts + numInfectiousContactsFromAges[ageInf]
#      Now, given that we expect totalNewInfectionContacts infectious contacts into our age category, how much overlap do we expect?
#       and how many are with susceptible individuals? 
            totalInAge = getTotalInAge(currentInternalStateDict, age)
#       Now when we draw totalNewInfectionContacts from totalInAge with replacement, how many do we expect?
#       For now, a simplifying assumption that there are *many more* individuals in totalInAge than there are   totalNewInfectionContacts
#       So we don't have to deal with multiple infections for the same individual.  TODO - address in future code update, raise issue for this
            if totalInAge > 0.0:
                numNewInfected = totalNewInfectionContacts*(numSuscept/totalInAge)
            else:
                numNewInfected = 0.0
            newInfectedsByAge[age] = numNewInfected
    return newInfectedsByAge


# CurrentlyInUse        
def getInternalInfection(dictOfStates, ageMixingInfectionMatrix, time):
    infectionsByNode = {}

    for node in dictOfStates[time]:
        infectionsByNode[node] = doInternalInfectionProcess(dictOfStates[time][node], ageMixingInfectionMatrix)

    return infectionsByNode


# CurrentlyInUse
# internalStateDict should have keys like (age, compartment)
#  The values are number of people in that node in that state
#  diseaseProgressionProbs should have outward probabilities per timestep (rates)
#  So we will need to do some accounting
def internalStateDiseaseUpdate(currentInternalStateDict, diseaseProgressionProbs):
    newStates = {}
    for (age, state), people in currentInternalStateDict.items():
        outTransitions = diseaseProgressionProbs[age].get(state, {})
        for nextState in outTransitions:
            newStates.setdefault((age, nextState), 0.0)
            newStates[(age, nextState)] += outTransitions[nextState] * people
    return newStates


# CurrentlyInUse
def getInternalProgressionAllNodes(currStates, diseaseProgressionProbs):
    """
    Given the values in the current state in the regions, progress the disease based on the progression matrix. This
    method is
    :param currStates: The current state for every region is not modified
    :param diseaseProgressionProbs: matrix with the probability of each transition
    """
    progression = {}
    for regionID, currRegion in currStates.items():
        progression[regionID] = internalStateDiseaseUpdate(currRegion, diseaseProgressionProbs)
    return progression


def mergeExposed(*args):
    exposedTotal = {}
    for infectionsDict in args:
        for regionID, region in infectionsDict.items():
            exposedRegionTotal = exposedTotal.setdefault(regionID, {})
            for age, exposed in region.items():
                exposedRegionTotal[age] = exposedRegionTotal.setdefault(age, 0.0) + exposed
    return exposedTotal


def exposeRegions(infections, states):
    """
    :param infections: a dict with number of infections per region per age
    :param states: dict representing the slice of time we want to alter

    This function modifies the state parameter.
    """
    for regionID in infections:
        for age in infections[regionID]:
            expose(age, infections[regionID][age], states[regionID])


def modelStatesToPandas(timeseries: Dict[int, Dict[str, Dict[Tuple[str, str], float]]]) -> pd.DataFrame:
    """
    Takes an instance of NetworkOfPopulations.states and transforms it into a pandas DataFrame.

    :param timeseries: dict object with the state over the times
    :return: a pandas dataframe in tabular format with the following columns:
             - time
             - healthboard
             - age
             - state
             - total
    """
    rows = []
    for time, nodes in timeseries.items():
        for nodeID, node in nodes.items():
            for (age, state), value in node.items():
                rows.append({"time": time, "healthboard": nodeID, "age": age, "state": state, "total": value})
    return pd.DataFrame(rows)


def plotStates(df, healthboards=None, states=None, ncol=3, sharey=False, figsize=None, cmap=None):
    """
    Plots a grid of plots, one plot per node, filtered by disease progression states (each states will be a line). The
    graphs are all Number of People x Time

    :param df: pandas DataFrame with healthboard, time, state and total columns
    :param healthboards: creates one plot per health board listed (None means all health boards)
    :param states: plots one curve per state listed (None means all states)
    :param ncol: number of columns (the number of rows will be calculated to fit all graphs)
    :param sharey: set to true if all plots should have the same y-axis
    :param figsize: select the size of each individual plot
    :param cmap: color map to use
    :return: returns a matplotlib figure
    """
    if healthboards is None:
        healthboards = df.healthboard.unique().tolist()
    if states is None:
        states = df.state.unique().tolist()
    if cmap is None:
        cmap = ListedColormap(["#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999", "#E69F00"])
    nrow = math.ceil(len(healthboards) / ncol)
    if figsize is None:
        if nrow >= 3:
            figsize = (20, 20)
        elif nrow == 2:
            figsize = (20, 10)
        else:
            figsize = (20, 5)

    if not healthboards:
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
            if count < len(healthboards):
                node = healthboards[count]
                count += 1
                grouped = df[df.healthboard == node].groupby(["time", "state"]).sum()
                indexed = grouped.reset_index().pivot(index="time", columns="state", values="total")

                ax = axes[i, j]
                indexed.plot(ax=ax, legend=False, title=node, cmap=cmap)
                ax.set_ylabel("Number of People")
                ax.set_xlabel("Time")

    assert ax is not None, "ax was never assigned"
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    return fig


# The functions below are the only operations that need to know about the actual state values.
SUSCEPTIBLE_STATE = "S"
EXPOSED_STATE = "E"
INFECTIOUS_STATES = ["I", "A"]


class NetworkOfPopulation(NamedTuple):
    """
    This type has all the internal data used by this model
    """
    progression: Dict[str, Dict[str, Dict[str, float]]]
    states: Dict[int, Dict[str, Dict[Tuple[str, str], float]]]
    graph: nx.DiGraph
    infectionMatrix: loaders.MixingMatrix
    movementMultipliers: Dict[int, float]


def createNetworkOfPopulation(disasesProgressionFn, populationFn, graphFn, ageInfectionMatrixFn, movementMultipliersFn=None) -> NetworkOfPopulation:
    # disases progression matrix
    with open(disasesProgressionFn) as fp:
        progression = loaders.readCompartmentRatesByAge(fp)

    # Check some requirements for this particular model to work with the progression matrix
    for states in progression.values():
        assert SUSCEPTIBLE_STATE not in states, "progression from susceptible state is not allowed"
        for state, nextStates in states.items():
            for nextState in nextStates:
                assert state == nextState or nextState != EXPOSED_STATE, "progression into exposed state is not allowed other than in self reference"

    # people movement's graph
    graph = loaders.genGraphFromContactFile(graphFn)

    # movement multipliers (dampening or heightening)
    if movementMultipliersFn is not None:
        with open(movementMultipliersFn) as fp:
            movementMultipliers = loaders.readMovementMultipliers(fp)
    else:
        movementMultipliers = {}

    # age-based infection matrix
    infectionMatrix = loaders.MixingMatrix(ageInfectionMatrixFn)

    agesInInfectionMatrix = set(infectionMatrix)
    for age in infectionMatrix:
        assert agesInInfectionMatrix == set(infectionMatrix[age]), "infection matrix columns/rows mismatch"

    # population census data
    with open(populationFn) as fp:
        population = loaders.readPopulationAgeStructured(fp)

    # Checks across datasets
    assert agesInInfectionMatrix == set(progression.keys()), "infection matrix and progression ages mismatch"
    assert agesInInfectionMatrix == {age for region in population.values() for age in region}, "infection matrix and population ages mismatch"
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
        infectionMatrix=infectionMatrix,
        movementMultipliers=movementMultipliers,
    )


def createNextStep(progression, exposed, currState):
    """
    """
    assert progression.keys() == exposed.keys() == currState.keys(), "missing regions"

    nextStep = copy.deepcopy(currState)
    for region in nextStep.values():
        for (age, state) in region.keys():
            # We need to keep the susceptibles in order to infect them
            if state != SUSCEPTIBLE_STATE:
                region[(age, state)] = 0.0

    for regionID, region in progression.items():
        for (age, state), value in region.items():
            # Note that the progression is responsible for populating every other state
            assert state != SUSCEPTIBLE_STATE, "Susceptibles can't be part of progression states"
            nextStep[regionID][(age, state)] = value

    for regionID, region in exposed.items():
        for age, exposed in region.items():
            suscept = nextStep[regionID][(age, SUSCEPTIBLE_STATE)]
            # The probability one of our susceptables avoids all infection
            probOneAvoidsAll = (1-(1/suscept))**exposed
            # so the probability that it *does* get infected is (1-probOneAvoidsAll)
            # and because we're only calculating expectation here we can use
            # linearity of expectation to sum up over all susceptibles by multiplication
            # Note: this should *never* be more that exposed, and should tend toward
            # exposed as suscept gets very large
            modifiedExposed = suscept*(1-probOneAvoidsAll)
            expose(age, modifiedExposed, nextStep[regionID])

    return nextStep


def getSusceptibles(age, currentInternalStateDict):
    return currentInternalStateDict[(age, SUSCEPTIBLE_STATE)]


def getInfectious(age, currentInternalStateDict):
    total = 0.0
    for state in INFECTIOUS_STATES:
        total += currentInternalStateDict[(age, state)]
    return total


def expose(age, exposed, region):
    """
    :param age: age group that will be exposed
    :param exposed: number (float) of exposed individuals
    :param region: dict representing a region, with all the (age, state) tuples

    This function modifies the region in-place, removing people from susceptible and adding them to exposed
    """
    assert region[(age, SUSCEPTIBLE_STATE)] >= exposed, f"S:{region[(age, SUSCEPTIBLE_STATE)]} < E:{exposed}"

    region[(age, EXPOSED_STATE)] += exposed
    region[(age, SUSCEPTIBLE_STATE)] -= exposed

# NotCurrentlyInUse
def nodeUpdate(graph, dictOfStates, time, headString):
        print('\n\n===== BEGIN update 1 at time ' + str(time) + '=========' + headString)
        for node in list(graph.nodes()):
             print('Node ' + str(node)+ " E-A-I at mature " + str(dictOfStates[time][node][('m', 'E')]) + " " +str(dictOfStates[time][node][('m', 'A')]) + " " + str(dictOfStates[time][node][('m', 'I')]))
        print('===== END update 1 at time ' + str(time) + '=========')
