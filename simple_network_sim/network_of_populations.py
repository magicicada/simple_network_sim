import copy

from collections import namedtuple

from simple_network_sim import loaders


# NotCurrentlyInUse
# This needs amendment to have different node populations and age structure
# Right now it is a framework function, to allow ongoing dev - uniform age structure in each,
def doSetupAgeStruct(G, dictOfStates, numInside, ages, states):
    dictOfStates[0] = {}
    for guy in G.nodes():  
        internalState = {}
        for age in ages:
            for state in states:
                internalState[(age, state)] = 0
            internalState[(age, 'S')] = numInside
        dictOfStates[0][guy] = internalState
    return dictOfStates


# CurrentlyInUse
def countInfectionsAgeStructured(dictOfStates, time):
    total = 0
    for node in dictOfStates[time]:
        for (age, state) in dictOfStates[time][node]:
            if state == 'A' or state == 'I':
                total = total + dictOfStates[time][node][(age, state)]
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
           
    # print(dictOfStringsByNodeAndState)
    
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

    for time in range(timeHorizon):
        # make sure the next time exists, so that we can add exposed individuals to it
        nextTime = time + 1
        if nextTime not in network.states:
            network.states[nextTime] = copy.deepcopy(network.states[time])
            for region in network.states[nextTime].values():
                for state in region.keys():
                    region[state] = 0.0

        doInternalProgressionAllNodes(network.states, time, network.progression)

        doInteralInfectionProcessAllNodes(network.states, network.infectionMatrix, time)

        doBetweenInfectionAgeStructured(network.graph, network.states, time)

        timeSeriesInfection.append(countInfectionsAgeStructured(network.states, time))

    return timeSeriesInfection


# NotCurrentlyInUse
def nodeUpdate(graph, dictOfStates, time, headString):
        print('\n\n===== BEGIN update 1 at time ' + str(time) + '=========' + headString)
        for node in list(graph.nodes()):
             print('Node ' + str(node)+ " E-A-I at mature " + str(dictOfStates[time][node][('m', 'E')]) + " " +str(dictOfStates[time][node][('m', 'A')]) + " " + str(dictOfStates[time][node][('m', 'I')]))
        print('===== END update 1 at time ' + str(time) + '=========')


# CurrentlyInUse
def totalIndividuals(nodeState):
    return sum(nodeState.values())


# CurrentlyInUse
def getTotalInAge(nodeState, ageTest):
    total= 0
    for (age, state) in nodeState:
            if age == ageTest:
                total = total + nodeState[(age, state)]
    return total


# CurrentlyInUse
def getTotalInfected(nodeState):
    totalInfectedHere = 0
    for (age, state) in nodeState:
            if state == 'A' or state == 'I':
                totalInfectedHere = totalInfectedHere + nodeState[(age, state)]
    return totalInfectedHere


# CurrentlyInUse
def getTotalSuscept(nodeState):
    totalSusHere = 0
    for (age, state) in nodeState:
                if state == 'S':
                        totalSusHere = totalSusHere + nodeState[(age, state)]
    return totalSusHere


# CurrentlyInUse
# fractional people will come out of this
# right now this infects uniformly across age class by number of susceptibles in age class 
def distributeInfections(nodeState, newInfections):
    ageToSus = {}
    newInfectionsByAge = {}
    for (age, state) in nodeState:
        if state == 'S':
            ageToSus[age] = nodeState[(age, state)]
    totalSus = sum(ageToSus.values())
    if totalSus<newInfections:
        print('ERROR: Too many infections to distribute amongst age classes - adjusting num infections')
        newInfections = totalSus
    for age in ageToSus:
        if totalSus > 0:
           newInfectionsByAge[age] = (float(ageToSus[age])/float(totalSus))*newInfections
        else:
            newInfectionsByAge[age] = 0
    return newInfectionsByAge


def doIncomingInfectionsByNode(graph, currentState):
    totalIncomingInfectionsByNode = {}
    for receivingVertex in currentState:
        totalSusceptHere = getTotalSuscept(currentState[receivingVertex])
        totalIncomingInfectionsByNode[receivingVertex] = 0
        if totalSusceptHere > 0:
            neighbours = list(graph.predecessors(receivingVertex))
            for givingVertex in neighbours:
                if givingVertex == receivingVertex:
                    continue
                totalInfectedGiving = getTotalInfected(currentState[givingVertex])
                if totalInfectedGiving > 0:
                    weight = 1.0
                    if 'weight' not in graph[givingVertex][receivingVertex]:
                        print("ERROR: No weight available for edge " + str(givingVertex) + "," + str(
                            receivingVertex) + " assigning weight 1.0")
                    else:
                        weight = graph[givingVertex][receivingVertex]['weight']

                    fractionGivingInfected = totalInfectedGiving / totalIndividuals(
                        currentState[givingVertex])
                    fractionReceivingSus = totalSusceptHere / totalIndividuals(
                        currentState[receivingVertex])
                    totalIncomingInfectionsByNode[receivingVertex] = totalIncomingInfectionsByNode[
                                                                         receivingVertex] + weight * fractionGivingInfected * fractionReceivingSus

    return totalIncomingInfectionsByNode


# CurrentlyInUse
# To bring this in line with the within-node infection updates (and fix a few bugs), I'm going to rework
# it so that we calculate an *expected number* of infectious contacts more directly. Then we'll distribute and
# overlap them using the same infrastructure code that we'll use for the internal version, when we add that
# Reminder: I expect the weighted edges to be the number of *expected infectious* contacts (if the giver is infectious)
#  We may need to multiply movement numbers by a probability of infection to achieve this.   
def doBetweenInfectionAgeStructured(graph, dictOfStates, currentTime):
    totalIncomingInfectionsByNode = doIncomingInfectionsByNode(graph, dictOfStates[currentTime])
                        
    # This might over-infect - we will need to adapt for multiple infections on a single individual if we have high infection threat.  TODO raise an issue
    for vertex in totalIncomingInfectionsByNode:
        totalDelta = totalIncomingInfectionsByNode[vertex]
        deltaByAge = distributeInfections(dictOfStates[currentTime][vertex], totalDelta)
        for age in deltaByAge:
           assert dictOfStates[currentTime+1][vertex][(age, 'S')] >= deltaByAge[age], "number of infected cannot be greater than susceptible"
           dictOfStates[currentTime+1][vertex][(age, 'S')] = dictOfStates[currentTime+1][vertex][(age, 'S')] - deltaByAge[age]
           dictOfStates[currentTime+1][vertex][(age, 'E')] = dictOfStates[currentTime+1][vertex][(age, 'E')] + deltaByAge[age]



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

        numSuscept = currentInternalStateDict[(age, 'S')]
        if numSuscept>0:
            numInfectiousContactsFromAges = {}
            totalNewInfectionContacts = 0
            for ageInf in ageMixingInfectionMatrix[age]:
                totalInfectious = currentInternalStateDict[(ageInf, 'I')] + currentInternalStateDict[(ageInf, 'A')]
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
def doInteralInfectionProcessAllNodes(dictOfStates, ageMixingInfectionMatrix, time):
    nextTime = time+1
    for node in dictOfStates[time]:
            newInfected = doInternalInfectionProcess(dictOfStates[time][node], ageMixingInfectionMatrix)
            for age in newInfected:
                assert newInfected[age] <= dictOfStates[nextTime][node][(age, 'S')], f"More infected people than susceptible ({age}, {time}, {dictOfStates[nextTime][node][(age, 'S')]}, {newInfected[age]})"
                dictOfStates[nextTime][node][(age, 'E')] = dictOfStates[nextTime][node][(age, 'E')] + newInfected[age]
                dictOfStates[nextTime][node][(age, 'S')] = dictOfStates[nextTime][node][(age, 'S')] - newInfected[age]


# CurrentlyInUse
# internalStateDict should have keys like (age, compartment) 
#  The values are number of people in that node in that state
#  diseaseProgressionProbs should have outward probabilities per timestep (rates)
#  So we will need to do some accounting
def internalStateDiseaseUpdate(currentInternalStateDict, diseaseProgressionProbs):
    dictOfNewStates = {}
    for (age, state) in currentInternalStateDict:
        dictOfNewStates[(age, state)] = 0
    for (age, state) in currentInternalStateDict:
        if state == 'S':
            dictOfNewStates[(age, state)] = currentInternalStateDict[(age, state)]
        else:
            outTransitions = diseaseProgressionProbs[age][state]
            numberInPrevState = currentInternalStateDict[(age, state)]
    #         we're going to have non-integer numbers of people for now
            for nextState in outTransitions:
                numberInNext = outTransitions[nextState]*currentInternalStateDict[(age, state)]
                dictOfNewStates[(age, nextState)] = dictOfNewStates[(age, nextState)]  + numberInNext
    return dictOfNewStates


# CurrentlyInUse
def doInternalProgressionAllNodes(dictOfNodeInternalStates, currentTime, diseaseProgressionProbs):
    nextTime = currentTime +1
    currStates = dictOfNodeInternalStates[currentTime]
    if nextTime not in dictOfNodeInternalStates:
        dictOfNodeInternalStates[nextTime] = {}
    for vertex in currStates:
        nextProgressionState = internalStateDiseaseUpdate(currStates[vertex], diseaseProgressionProbs)
        dictOfNodeInternalStates[nextTime][vertex] = nextProgressionState


NetworkOfPopulation = namedtuple("NetworkOfPopulation", ["progression", "states", "graph", "infectionMatrix"])

def createNetworkOfPopulation(disasesProgressionFn, populationFn, graphFn):
    with open(disasesProgressionFn) as fp:
        progression = loaders.readCompartmentRatesByAge(fp)
    with open(populationFn) as fp:
        population = loaders.readPopulationAgeStructured(fp)
    graph = loaders.genGraphFromContactFile(graphFn)

    # TODO: read this from a file
    infectionMatrix = {}
    for ageA in progression.keys():
        for ageB in progression.keys():
            infectionMatrix.setdefault(ageA, {})[ageB] = 0.2

    state0 = {}
    for node in list(graph.nodes()):
        region = state0.setdefault(node, {})
        for age, compartments in progression.items():
            region[(age, "S")] = population[node][age]
            for compartment in compartments:
                region[(age, compartment)] = 0

    return NetworkOfPopulation(progression=progression, graph=graph, states={0: state0}, infectionMatrix=infectionMatrix)


def exposeRegions(regions, exposed, ageDistribution, state):
    """
    :param regions: a list with all regions to be infected
    :param exposed: number (float) of people that will be exposed
    :param ageDistribution: a dict of type {age: float} with the probabilities for exposition in each age
    :param state: dict representing the slice of time we want to alter

    This function modifies the state parameter.
    """
    assert sum(ageDistribution.values()) == 1.0, "the age distribution must add up to 1.0"
    for region in regions:
        for age, prob in ageDistribution.items():
            expose(age, exposed * prob, state[region])


def expose(age, exposed, region):
    """
    :param age: age group that will be exposed
    :param exposed: number (float) of exposed individuals
    :param region: dict representing a region, with all the (age, state) tuples

    This function modifies the region in-place, removing people from susceptible and adding them to exposed
    """
    assert region[(age, "S")] >= exposed, "cannot expose more than number of susceptible"
    region[(age, "E")] += exposed
    region[(age, "S")] -= exposed
