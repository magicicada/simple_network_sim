import networkx as nx
import random
import sys
from collections import Counter
import matplotlib.pyplot as plt
import json


compartments = {}
compNames = ['S', 'E', 'A', 'I', 'H', 'R', 'D']


def doSetup(G, dictOfStates):
    dictOfStates[0] = {}
    for guy in G.nodes():
        dictOfStates[0][guy] = 'S'
        
#  This needs amendment to have different node populations and age structure
#  Right now it is a framework function, to allow ongoing dev - uniform age structure in each, 
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

# making this general to include arbitrary future attributes.  Location is the primary one for right now
# keeps them in a dictionary and returns that.  Keys are  
def readNodeAttributesJSON(filename):
    f = open(filename,) 
    node_data = json.load(f)
    return node_data

# this could use some exception-catching (in fact, basically everything could)
# we're going to have a nested dictionary - age to dictionary of parameters
def readParametersAgeStructured(filename):
    agesDictionary = {}
    for line in open(filename, 'r'):
        split = line.strip().split(":")
        label = split[0].strip()
        agePar = label.split(",")
        age = agePar[0].strip()
        paramName = agePar[1].strip()
        if age not in agesDictionary:
            agesDictionary[age] = {}
        agesDictionary[age][paramName] = float(split[1].strip())
    return agesDictionary


# this could use some exception-catching (in fact, basically everything could)
def readParameters(filename):
    parametersDictionary = {}
    for line in open(filename, 'r'):
        split = line.strip().split(":")
        parametersDictionary[split[0].strip()] = float(split[1].strip())
    return parametersDictionary

def checkForParameters(dictOfParams, ageStructured):
    ageStructParams = ['e_escape_young', 'a_escape_young', 'a_to_i_young', 'a_to_r_young', 'i_escape_young', 'i_to_d_young', 'i_to_h_young', 'h_escape_young', 'h_to_d_young',
                       'e_escape_mature', 'a_escape_mature', 'a_to_i_mature', 'a_to_r_mature', 'i_escape_mature', 'i_to_d_mature', 'i_to_h_mature', 'h_escape_mature', 'h_to_d_mature',
                        'e_escape_old', 'a_escape_old', 'a_to_i_old', 'a_to_r_old', 'i_escape_old', 'i_to_d_old', 'i_to_h_old', 'h_escape_old', 'h_to_d_old']
    vanillaParams = ['e_escape', 'a_escape', 'a_to_i', 'i_escape', 'i_to_d', 'i_to_h', 'h_escape', 'h_to_d']
    
    if ageStructured:
        for param in ageStructParams:
            if param not in dictOfParams:
                print =("ERROR: missing parameter " + str(param))
                return False
    else:
        for param in vanillaParams:
            if param not in dictOfParams:
                print =("ERROR: missing parameter " + str(param))
                return False
    return True    


def setUpParametersVanilla(dictOfParams):
    fromStateTrans = {}
    fromStateTrans['E'] ={'E':1-dictOfParams['e_escape'], 'A': dictOfParams['e_escape']}
    fromStateTrans['A'] = {'A':1-dictOfParams['e_escape'], 'I': dictOfParams['e_escape']*dictOfParams['a_to_i'], 'R':dictOfParams['e_escape']*(1-dictOfParams['a_to_i'])}
    fromStateTrans['I'] =  {'I':1-dictOfParams['i_escape'], 'D': dictOfParams['i_escape']*dictOfParams['i_to_d'], 'H':dictOfParams['i_escape']*(dictOfParams['i_to_h']),
                             'R':dictOfParams['i_escape']*(1-dictOfParams['i_to_h'] -dictOfParams['i_to_d']) }
    fromStateTrans['H'] = {'H':1-dictOfParams['h_escape'], 'D': dictOfParams['h_escape']*dictOfParams['h_to_d'], 'R':dictOfParams['h_escape']*(1- dictOfParams['h_to_d'])}
    fromStateTrans['R'] = {'R':1.0}
    fromStateTrans['D'] = {'D':1.0}
    return fromStateTrans

def setUpParametersAges(dictByAge):
    ageToStateTrans = {}
    for age in dictByAge:
        ageToStateTrans[age] = {}
        ageToStateTrans[age] = setUpParametersVanilla(dictByAge[age])
    return ageToStateTrans
    

def chooseFromDistrib(distrib):
    sumSoFar = 0
    thisLuck = random.random()
    for value in distrib:
        sumSoFar = sumSoFar + distrib[value]
        if thisLuck <= sumSoFar:
            return value
    print('Something has gone wrong - no next state was returned.  Choosing arbitrarily')
    return min(distrib.keys())

# Simplest sensible model: no age classes, uniform transitions between states
# each vertex will have a state at each timestep 
def doProgression(dictOfStates, currentTime):
    # currentTime = max(dictOfStates.values())
    nextTime = currentTime +1
    currStates = dictOfStates[currentTime]
    dictOfStates[nextTime] = {}
    
    for vertex in currStates:
       # get the state, then then possibilities
       state = currStates[vertex]
       if state =='R' or state == 'D':
           dictOfStates[nextTime][vertex] = state
       elif state != 'S': 
           dictOfStates[nextTime][vertex] = chooseFromDistrib(fromStateTrans[state])
       else:
           dictOfStates[nextTime][vertex] = dictOfStates[currentTime][vertex]

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
        dictOfStates[currentTime+1][fella] = 'E'
    
def prettyPrint(dictOfStates, time):
    states = ['S']
    stateString = 'S,'
    for state in fromStateTrans:
        states.append(state)
        stateString = stateString + state + ","
    print(stateString)
    
    counts = Counter(dictOfStates[time].values())
    print(counts)
    
def countInfections(dictOfStates, time):
    counts = Counter(dictOfStates[time].values())
    sumBoth = 0
    if 'I' in counts:
       sumBoth = sumBoth + counts['I']
    if 'A' in counts:
        sumBoth = sumBoth + counts['A']
    return sumBoth

def countInfectionsAgeStructured(dictOfStates, time):
    total = 0
    for node in dictOfStates[time]:
        for (age, state) in dictOfStates[time][node]:
            if state == 'A' or state == 'I':
                total = total + dictOfStates[time][node][(age, state)]
    return total

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

def basicSimulationInternalAgeStructure(graph, numInfected, timeHorizon, genericInfection, ageInfectionMatrix, diseaseProgressionProbs):
    print('WARNING - FUNCTION NOT PROPERLY TESTED YET - basicSimulationInternalAgeStructure')
    ages = list(ageInfectionMatrix.values())
    timeSeriesInfection = []
    
    dictOfStates = {}
    numInside = 100
    ages = ['y', 'm', 'o']
    states = ['S', 'E', 'A', 'I', 'H', 'R', 'D']
    
    doSetupAgeStruct(graph, dictOfStates, numInside, ages, states)

    # for now, we choose a random node and infect numInfected mature individuals - right now they are extra individuals, not removed from the susceptible class
    infectedNode = random.choices(list(graph.nodes()), k=1)
    for vertex in infectedNode:
        dictOfStates[0][vertex][('m', 'I')] = numInfected 
    
    for time in range(timeHorizon):
        doInternalProgressionAllNodes(dictOfStates, time, diseaseProgressionProbs)
        for node in dictOfStates[time]:
           doInternalInfectionProcess(dictOfStates[time][node], ageInfectionMatrix, ages, time)
        doBetweenInfectionAgeStructured(graph, dictOfStates, time, genericInfection)
        timeSeriesInfection.append(countInfectionsAgeStructured(dictOfStates, time))
        
        print('\n\n===== BEGIN update at time ' + str(time) + '=========')
        for node in list(graph.nodes()):
            print('Node ' + str(node))
            print(dictOfStates[time][node])
        print('===== END update at time ' + str(time) + '=========')
        
    return timeSeriesInfection

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

def generateHouseholdsAggregateGraph(numHouseholds, radius):
    # generate a random geometric graph for households in range:
    randomGeometric = nx.random_geometric_graph(numHouseholds, radius)
    return randomGeometric

def addIllicitEdges(existingGraph, numberEdges):
    for i in range(numberEdges):
        listOfTwo = list(random.sample(list(existingGraph.nodes()), 2))
        existingGraph.add_edge(listOfTwo[0], listOfTwo[1])
        
def generateIllicitEdges(existingGraph, numberEdges):
    newEdges = []
    for i in range(numberEdges):
        listOfTwo = list(random.sample(list(existingGraph.nodes()), 2))
        existingGraph.add_edge(listOfTwo[0], listOfTwo[1])
        newEdges.append((listOfTwo[0], listOfTwo[1]))
    return newEdges
        
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
        adds = random.sample(remainingChoice, numInEach -1)
        inAGroup.append(adds)

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
        adds = random.sample(remainingChoice, min(sizeGroups -1, len(remainingChoice)))
        inAGroup.append(adds)
        for item in adds:
            strongEdges.append((start, item))
            strongEdges.append((item, start))
            for item2 in adds:
                strongEdges.append((item2, item))
                strongEdges.append((item, item2))

    return strongEdges
    
def generateMeanPlot(listOfPlots):
    meanForPlot = []
    # print(listOfPlots)
    for i in range(len(listOfPlots[0])):
        sumTot = 0
        for j in range(len(listOfPlots)):
            sumTot = sumTot + listOfPlots[j][i]
        meanForPlot.append(float(sumTot)/len(listOfPlots))
    return meanForPlot
        
 # Internal states for nodes -
#   The plan is that each node can have an associated dictionary that gives internal compartments
#   (in the disease compartment sense)
#   changes we will need to make:
#   - add internal disease progression function
#   - add setup of internal states
#   - add capacity to read populations
#   - do we want age structure?
#   - change the basic simulation to use the right disease update infrastructure
#   - alter the network infectious process to take the internal state into account


# Done make disease transmission internal update
# rework overall simulation to use internal-compartment versions
# write plotting function for node selection
# run a single-node version and do plots
# run a two-vertex version with plots
# run a path version with plots, different population sizes


def totalIndividuals(nodeState):
    return sum(nodeState.values())

def getTotalInfected(nodeState):
    totalInfectedHere = 0
    for (age, state) in nodeState:
            if state == 'A' or state == 'I':
                totalInfectedHere = totalInfectedHere + nodeState[(age, state)]
    return totalInfectedHere

def getTotalSuscept(nodeState):
    totalSusHere = 0
    for (age, state) in nodeState:
                if state == 'S':
                        totalSusHere = totalSusHere + nodeState[(age, state)]
    return totalSusHere

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
    

#  This function will need improving from a a modelling standpoint.
#  it will be some function of the number of I/A in nodeState1, S in nodeState2, and the weight
# of the edge between the two.
# should give a float between 0 and 1 that is the probability that a S in node2 is infected by a migrant from node1 
def fractionInfectedByEdge(nodeState1, nodeState2, edgeWeight):
    fractionInfectedSource = getTotalInfected(nodeState1)/totalIndividuals(nodeState1)
    fractionSusceptibleDest = getTotalSuscept(nodeState2)/totalIndividuals(nodeState2) 
    return fractionInfectedSource*fractionSusceptibleDest*edgeWeight

def doBetweenInfectionAgeStructured(graph, dictOfStates, currentTime, genericInfectionProb):
#   This dictionary should have nodes as keys, floats as values that are the probability of escaping infection from other nodes
    avoidInfection = {}
    for node in list(graph.nodes()):
        avoidInfection[node] = 1.0

    newInfectedPressures = {}
    
    for vertex in dictOfStates[currentTime]:
        totalInfectedHere = getTotalInfected(dictOfStates[currentTime][vertex])
        if totalInfectedHere >0:
            neighbours = list(graph.neighbors(vertex))
            for neigh in neighbours:
                    totalSusHere = getTotalSuscept(dictOfStates[currentTime][vertex])
                    if totalSusHere >0:
                        if 'weight' not in graph[vertex][neigh]:
                            probabilityOfInfection = genericInfectionProb
                        else:
                            probabilityOfInfection = graph[vertex][neigh]['weight']
                        avoidInfection[neigh] = avoidInfection[neigh]*(1-fractionInfectedByEdge(dictOfStates[currentTime][vertex], dictOfStates[currentTime][neigh], probabilityOfInfection))
    for vertex in avoidInfection:
        total_delta = (1-avoidInfection[vertex])*getTotalSuscept(dictOfStates[currentTime][vertex])
        deltaByAge = distributeInfections(dictOfStates[currentTime][vertex], total_delta)
        for age in deltaByAge:
           dictOfStates[currentTime+1][vertex][(age, 'S')] = dictOfStates[currentTime+1][vertex][(age, 'S')] - deltaByAge[age]
           dictOfStates[currentTime+1][vertex][(age, 'A')] = dictOfStates[currentTime+1][vertex][(age, 'A')] + deltaByAge[age]

#  the parameter ageMixingInfectionMatrix should include mixing information that incorporates
#  probability of infection as well - that is the entry at row age1 column age2
#  is the rate of contact from age1 to age2 of *infectious contact* - e.g.
#  if we expect half of contacts from young to mature to be infectious, and 0.25 of all possible young to mature contacts happen
# (so the expected number of contacts from young to old is (number_young)*(number_old)*0.25), then the entry in this matrix
# should be 0.125.  Note that it need not be symmetric.
# concern: need to think carefully about this asymmetry.  For now, I'll be using a uniform infectiousness
# by contact to generate that matrix 
def doInternalInfectionProcess(currentInternalStateDict, ageMixingInfectionMatrix, ages, time):
    newInfectedsByAge = {}
    for age in ages:
        newInfectedsByAge[age] = 0
        # print('\n\n\n')
        # print(currentInternalStateDict)
        numSuscept = currentInternalStateDict[(age, 'S')]
        if numSuscept>0:
            numInfectiousContactsFromAges = {}
            for ageInf in ages:
                totalInfectious = currentInternalStateDict[(ageInf, 'I')] + currentInternalStateDict[(ageInf, 'A')]
                numInfectiousContactsFromAges[ageInf] = totalInfectious*numSuscept*ageMixingInfectionMatrix[ageInf][age]
            totalAvoid = 1.0
            for numInf in list(numInfectiousContactsFromAges.values()):
                totalAvoid = totalAvoid*(1-float(numInf)/float(numSuscept))
            numNewInfected = (1-totalAvoid)*numSuscept
            newInfectedsByAge[age] = numNewInfected
    return newInfectedsByAge
        
        


# internalStateDict should have keys like (age, compartment) 
#  The values are number of people in that node in that state
#  diseaseProgressionProbs should have outward probabilities per timestep (rates)
#  So we will need to do some accounting
def internalStateDiseaseUpdate(currentInternalStateDict, diseaseProgressionProbs):
    dictOfNewStates = {}
    for (age, state) in currentInternalStateDict:
        if state =='R' or state == 'D':
            dictOfNewStates[(age, state)] = currentInternalStateDict[(age, state)]
        else:
            dictOfNewStates[(age, state)] = 0
    for (age, compartment) in currentInternalStateDict:
        if compartment == 'S':
            dictOfNewStates[(age, 'S')] = currentInternalStateDict[(age, 'S')]
        else:
            # print("\n\n\n========diseaseProgressionProbs====")
            # print(diseaseProgressionProbs['y'])
            outTransitions = diseaseProgressionProbs[age][compartment]
            numberInPrevState = currentInternalStateDict[(age, compartment)]
    #         we're going to have non-integer numbers of people for now
            for nextState in outTransitions:
                numberInNext = outTransitions[nextState]*currentInternalStateDict[(age, compartment)]
                dictOfNewStates[(age, nextState)] = dictOfNewStates[(age, nextState)]  + numberInNext
        
    return dictOfNewStates

def doInternalProgressionAllNodes(dictOfNodeInternalStates, currentTime, diseaseProgressionProbs):
    nextTime = currentTime +1
    currStates = dictOfNodeInternalStates[currentTime]
    dictOfNodeInternalStates[nextTime] = {}
    for vertex in currStates:
        nextProgressionState = internalStateDiseaseUpdate(currStates[vertex], diseaseProgressionProbs)
        dictOfNodeInternalStates[nextTime][vertex] = nextProgressionState
        
        
# This is a strawman version of this function for testing
def setupInternalPopulations(graph, listOfStates, ages):
    dictOfInternalStates = {}
    dictOfInternalStates[0] = {}
    currentTime = 0
    for node in list(graph.nodes()):
        dictOfInternalStates[0][node] = {}
        for state in listOfStates:
            for age in ages:
                dictOfInternalStates[0][node][(age, state)] = 0
        for age in ages:
            dictOfInternalStates[0][node][(age, 'S')] = 20
        
    randomNode = random.choice(list(graph.nodes()))
#     Note the arbitrary age seed, and all are asymptomatic, and number is fixed and arbitrary
    dictOfInternalStates[0][node][(ages[0], 'A')] = 5
    
    return dictOfInternalStates    
    
    
 #  A bit of sample model operation.     
    
    
# 
# print('drawing the graphs')
# nx.draw(baseGraph, pos, node_size=8, alpha = 0.3, color = 'dodgeblue')
# nx.draw_networkx_edges(baseGraph, pos, edgelist=strongEdges, edge_color = 'green', width=2.0)
# nx.draw_networkx_edges(baseGraph, pos, edgelist=longDistance, edge_color = 'maroon', width=1.0)
# plt.show()
# houseMembers = {}
# nearHouses = {}
# numGroups = 50
# sizeGroups = 6
baseWeight = 0.5
# strongWeight = 0.8
ages=['y', 'm', 'o']
numInfected = 10
genericInfection = 0.4
ageInfectionMatrix = {}
contactRate = 0.2
for age in ages:
    ageInfectionMatrix[age]  = {}
ageInfectionMatrix['y']['y'] = contactRate
ageInfectionMatrix['y']['m'] = contactRate
ageInfectionMatrix['y']['o']= contactRate
ageInfectionMatrix['m']['y']= contactRate
ageInfectionMatrix['m']['m']= contactRate
ageInfectionMatrix['m']['o']= contactRate
ageInfectionMatrix['o']['y']= contactRate
ageInfectionMatrix['o']['m']= contactRate
ageInfectionMatrix['o']['o']= contactRate

# 
# baseGraph = generateHouseholdsAggregateGraph(10, 0.06)
#
baseGraph = nx.path_graph(10)
for (u, v) in list(baseGraph.edges()):
     baseGraph[u][v]['weight'] = baseWeight
#     
states = setupInternalPopulations(baseGraph, compNames, ages)
# for node in states[0]:
#     print(states[0][node])

params = readParametersAgeStructured(sys.argv[1])
for guy in params:
    print (params[guy])

ageToTrans = setUpParametersAges(params)
for age in ageToTrans:
    print (age)
    print( ageToTrans[age])



# pos = nx.get_node_attributes(baseGraph, 'pos')
# strongEdges = generateChildcareEdgesAggregate(baseGraph, numGroups, sizeGroups)
# for (u, v) in strongEdges:
#     baseGraph.add_edge(u, v)
#     baseGraph[u][v]['weight'] = strongWeight
# 
# baseGraph.add_edges_from(strongEdges)
# 
# longDistance = generateIllicitEdges(baseGraph, 100)
# 
# paraDict = readParameters(sys.argv[1])
# fromStateTrans = setUpParametersVanilla(paraDict)
# print(fromStateTrans)
# 
basicPlots = []
# withGroups = []
# withIllicit = []
time = 10
numTrials = 1
# for i in range(numTrials): 
#     withGroups.append(basicSimulation(baseGraph, 4, time, 0.1))
# print('Done withGroups')
for i in range(numTrials):
     basicPlots.append(basicSimulationInternalAgeStructure(baseGraph, numInfected, time, genericInfection, ageInfectionMatrix, ageToTrans))
# # print('Done basic')
# addIllicitEdges(baseGraph, sizeGroups^2*numGroups)
# for i in range(numTrials):
#     withIllicit.append(basicSimulation(baseGraph, 4, time, 0.1))
# print('Done withIllicit')
# 
# plt.plot(generateMeanPlot(withIllicit), color='maroon', label = 'with_illicit')
# plt.plot(generateMeanPlot(withGroups), color='green', label = 'with_groups')
plt.plot(generateMeanPlot(basicPlots), color = 'dodgerblue', label='basic')
# plt.legend()
# # plt.ylim(top=200)
# 
plt.savefig('withAges.pdf')
