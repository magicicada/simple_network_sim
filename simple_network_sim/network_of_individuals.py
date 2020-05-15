import random
from collections import Counter

import networkx as nx


# NotCurrentlyInUseByCoreModel
def doSetup(G, dictOfStates):
    dictOfStates[0] = {}
    for guy in G.nodes():
        dictOfStates[0][guy] = 'S'


# NotCurrentlyInUse
def chooseFromDistrib(distrib):
    sumSoFar = 0
    thisLuck = random.random()
    for value in distrib:
        sumSoFar = sumSoFar + distrib[value]
        if thisLuck <= sumSoFar:
            return value
    print('Something has gone wrong - no next state was returned.  Choosing arbitrarily')
    return min(distrib.keys())


# NotCurrentlyInUseByCoreModel
def readParameters(filename):
    parametersDictionary = {}
    try:
        for line in open(filename, 'r'):
            split = line.strip().split(":")
            parametersDictionary[split[0].strip()] = float(split[1].strip())
    except IndexError:
        raise Exception(f"Error: Malformed input \"{line.rstrip()}\" in {filename}") from None
    return parametersDictionary


# NotCurrentlyInUseByCoreModel
def checkForParameters(dictOfParams, ageStructured):
    ageStructParams = ['e_escape_young', 'a_escape_young', 'a_to_i_young', 'a_to_r_young', 'i_escape_young',
                       'i_to_d_young', 'i_to_h_young', 'h_escape_young', 'h_to_d_young',
                       'e_escape_mature', 'a_escape_mature', 'a_to_i_mature', 'a_to_r_mature', 'i_escape_mature',
                       'i_to_d_mature', 'i_to_h_mature', 'h_escape_mature', 'h_to_d_mature',
                       'e_escape_old', 'a_escape_old', 'a_to_i_old', 'a_to_r_old', 'i_escape_old', 'i_to_d_old',
                       'i_to_h_old', 'h_escape_old', 'h_to_d_old']
    vanillaParams = ['e_escape', 'a_escape', 'a_to_i', 'i_escape', 'i_to_d', 'i_to_h', 'h_escape', 'h_to_d']

    if ageStructured:
        for param in ageStructParams:
            if param not in dictOfParams:
                print = ("ERROR: missing parameter " + str(param))
                return False
    else:
        for param in vanillaParams:
            if param not in dictOfParams:
                print = ("ERROR: missing parameter " + str(param))
                return False
    return True


# NotCurrentlyInUseByCoreModel
# Simplest sensible model: no age classes, uniform transitions between states
# each vertex will have a state at each timestep
def doProgression(dictOfStates, currentTime):
    # currentTime = max(dictOfStates.values())
    nextTime = currentTime + 1
    currStates = dictOfStates[currentTime]
    dictOfStates[nextTime] = {}

    for vertex in currStates:
        # get the state, then then possibilities
        state = currStates[vertex]
        if state == 'R' or state == 'D':
            dictOfStates[nextTime][vertex] = state
        elif state != 'S':
            dictOfStates[nextTime][vertex] = chooseFromDistrib(fromStateTrans[state])
        else:
            dictOfStates[nextTime][vertex] = dictOfStates[currentTime][vertex]


# NotCurrentlyInUseByCoreModel
def doInfection(graph, dictOfStates, currentTime, genericInfectionProb):
    newInfected = []
    for vertex in dictOfStates[currentTime]:

        if dictOfStates[currentTime][vertex] == 'I' or dictOfStates[currentTime][vertex] == 'A':
            neighbours = list(graph.neighbors(vertex))
            for neigh in neighbours:
                if dictOfStates[currentTime][neigh] == 'S':
                    if 'weight' not in graph[vertex][neigh]:
                        probabilityOfInfection = genericInfectionProb
                    else:
                        probabilityOfInfection = graph[vertex][neigh]['weight']
                    thisLuck = random.random()
                    if thisLuck <= probabilityOfInfection:
                        newInfected.append(neigh)
    for fella in newInfected:
        dictOfStates[currentTime + 1][fella] = 'E'


# NotCurrentlyInUseByCoreModel
def prettyPrint(dictOfStates, time):
    states = ['S']
    stateString = 'S,'
    for state in fromStateTrans:
        states.append(state)
        stateString = stateString + state + ","
    print(stateString)

    counts = Counter(dictOfStates[time].values())
    print(counts)


# NotCurrentlyInUseByCoreModel
def countInfections(dictOfStates, time):
    counts = Counter(dictOfStates[time].values())
    sumBoth = 0
    if 'I' in counts:
        sumBoth = sumBoth + counts['I']
    if 'A' in counts:
        sumBoth = sumBoth + counts['A']
    return sumBoth


# NotCurrentlyInUseByCoreModel
def basicSimulation(graph, numInfected, timeHorizon, genericInfection):
    timeSeriesInfection = []
    # choose a random set of initially infected
    infected = random.choices(list(graph.nodes()), k=numInfected)
    print(infected)
    dictOfStates = {}
    doSetup(graph, dictOfStates)
    for vertex in infected:
        dictOfStates[0][vertex] = 'I'

    for time in range(timeHorizon):
        doProgression(dictOfStates, time)
        doInfection(graph, dictOfStates, time, genericInfection)
        timeSeriesInfection.append(countInfections(dictOfStates, time))

    return timeSeriesInfection


# NotCurrentlyInUseByCoreModel
def generateHouseholds(numHouseholds, radius, locations, householdMembership, withinNeighbourhood):
    # generate a random geometric graph for households in range:
    randomGeometric = nx.random_geometric_graph(numHouseholds, radius)
    householdSizes = [1, 1, 2, 2, 2, 2, 3, 4, 4, 3, 3, 2]
    householdToMem = {}
    wholeGraph = nx.MultiGraph()
    for household in list(randomGeometric.nodes()):
        householdToMem[household] = []
        numMembers = random.choice(householdSizes)
        theMembers = []
        for member in range(numMembers):
            theMembers.append((household, member))
            wholeGraph.add_node((household, member))
            householdToMem[household].append((household, member))
        for member in theMembers:
            for secondMem in theMembers:
                if member != secondMem:
                    wholeGraph.add_edge(member, secondMem)
    for household in list(randomGeometric.nodes()):
        neighbourHouse = list(randomGeometric.neighbors(household))
        withinNeighbourhood[household] = neighbourHouse
        for neighbour in neighbourHouse:
            for fella in householdToMem[household]:
                for guy in householdToMem[neighbour]:
                    if (fella, guy) not in wholeGraph.edges():
                        wholeGraph.add_edge(fella, guy)
    householdMembership = householdToMem
    return wholeGraph


# NotCurrentlyInUseByCoreModel
def generateHouseholdsAggregateGraph(numHouseholds, radius):
    # generate a random geometric graph for households in range:
    randomGeometric = nx.random_geometric_graph(numHouseholds, radius)
    return randomGeometric


# NotCurrentlyInUseByCoreModel
def addIllicitEdges(existingGraph, numberEdges):
    for i in range(numberEdges):
        listOfTwo = list(random.sample(list(existingGraph.nodes()), 2))
        existingGraph.add_edge(listOfTwo[0], listOfTwo[1])


# NotCurrentlyInUseByCoreModel
def generateIllicitEdges(existingGraph, numberEdges):
    newEdges = []
    for i in range(numberEdges):
        listOfTwo = list(random.sample(list(existingGraph.nodes()), 2))
        existingGraph.add_edge(listOfTwo[0], listOfTwo[1])
        newEdges.append((listOfTwo[0], listOfTwo[1]))
    return newEdges


# NotCurrentlyInUseByCoreModel
# note function is not finished
# will eventually generate edges between and within households
def generateChildcareEdges(numInEach, numGroups, graph, householdWithin, nearHouseholds):
    print('WARNING - FUNCTION NOT PROPERLY FINISHED YET - generateChildcareEdges')
    inAGroup = []

    seeds = random.sample(list(graph.nodes()), numGroups)
    inAGroup.extend(seeds)

    for start in seeds:
        remainingChoice = []
        for fella in graph.nodes():
            if fella not in inAGroup:
                remainingChoice.append(fella)
        adds = random.sample(remainingChoice, numInEach - 1)
        inAGroup.append(adds)


# NotCurrentlyInUseByCoreModel
# for now all edges get the same weight
# The graph here is the geometric graph
def generateChildcareEdgesAggregate(graph, numGroups, sizeGroups):
    strongEdges = []
    inAGroup = []

    seeds = random.sample(list(graph.nodes()), numGroups)
    inAGroup.extend(seeds)
    for start in seeds:
        remainingChoice = []
        for fella in list(graph.neighbors(start)):
            if fella not in inAGroup:
                remainingChoice.append(fella)
        adds = random.sample(remainingChoice, min(sizeGroups - 1, len(remainingChoice)))
        inAGroup.append(adds)
        for item in adds:
            strongEdges.append((start, item))
            strongEdges.append((item, start))
            for item2 in adds:
                strongEdges.append((item2, item))
                strongEdges.append((item, item2))

    return strongEdges
