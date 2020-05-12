import networkx as nx
import random
import sys
from collections import Counter
import matplotlib.pyplot as plt
import json


compartments = {}
compNames = ['S', 'E', 'A', 'I', 'H', 'R', 'D']

# NotCurrentlyInUseByCoreModel
def doSetup(G, dictOfStates):
    dictOfStates[0] = {}
    for guy in G.nodes():
        dictOfStates[0][guy] = 'S'
        
# CurrentlyInUse         
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

# CurrentlyInUse 
# making this general to include arbitrary future attributes.  Location is the primary one for right now
# keeps them in a dictionary and returns that.  Keys are  
def readNodeAttributesJSON(filename):
    f = open(filename,) 
    node_data = json.load(f)
    return node_data

# CurrentlyInUse 
# this could use some exception-catching (in fact, basically everything could)
# we're going to have a nested dictionary - age to dictionary of parameters
def readParametersAgeStructured(filename):
    agesDictionary = {}
    try:
        for line in open(filename, 'r'):
            split = line.strip().split(":")
            label = split[0].strip()
            agePar = label.split(",")
            age = agePar[0].strip()
            paramName = agePar[1].strip()
            if age not in agesDictionary:
                agesDictionary[age] = {}
            agesDictionary[age][paramName] = float(split[1].strip())
    except IndexError:
        raise Exception(f"Error: Malformed input \"{line.rstrip()}\" in {filename}") from None
    checkAgeParameters(agesDictionary)
    return agesDictionary

# CurrentlyInUse
# This needs exception-catching, and probably shouldn't have hard-coded column indices. 
def readPopulationAgeStructured(filename):
    dictOfPops = {}
    boardInd =0
    sexInd = 1
    totalAgeInd = 2
    youngInd = 3
    matureInd = 4
    oldInd = 5
    with open(filename , 'r') as f:
       first_line = f.readline()
       for line in f:
          split = line.strip().split(",")
          board = split[boardInd]
          if board not in dictOfPops:
            dictOfPops[board] = {}
          sex = split[sexInd]
          if sex not in dictOfPops[board]:
                dictOfPops[board][sex] = {}
          numYoung = int(split[youngInd])
          numMature = int(split[matureInd])
          numOld = int(split[oldInd])
          numTotal = int(split[totalAgeInd])
          dictOfPops[board][sex]['y'] = numYoung
          dictOfPops[board][sex]['m'] = numMature
          dictOfPops[board][sex]['o'] = numOld
          dictOfPops[board][sex]['All_Ages'] = numTotal
          
#     a traversal to add in the totals
#     this is not great code, could be improved and made much more general - more robust against future age range changes
    for board in dictOfPops:
        numAllSex = 0
        numAllSexY = 0
        numAllSexM = 0
        numAllSexO = 0
        for sex in dictOfPops[board]:
            numAllSex = numAllSex + dictOfPops[board][sex]['All_Ages']
            numAllSexY = numAllSexY + dictOfPops[board][sex]['y']
            numAllSexM = numAllSexM + dictOfPops[board][sex]['m']
            numAllSexO = numAllSexO + dictOfPops[board][sex]['o']
        dictOfPops[board]['All_Sex'] = {}
        dictOfPops[board]['All_Sex']['y'] = numAllSexY
        dictOfPops[board]['All_Sex']['m'] = numAllSexM
        dictOfPops[board]['All_Sex']['o'] = numAllSexO
        dictOfPops[board]['All_Sex']['All_Ages'] = numAllSex
                                     
    return dictOfPops


# CurrentlyInUse
# at the moment this uses vanilla networkx edge list reading - needs weights
#  I've set it apart as its own function in case we want to do anything fancier with edge files
# in future - e.g. sampling, generating movements, whatever
# it should return a networkx graph, ideally with weighted edges
# eventual replacement with HDF5 reading code?
def genGraphFromContactFile(filename):
    G = nx.read_edgelist(filename, create_using=nx.DiGraph, delimiter=",", data=(('weight',float),))
    return G
    
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


# CurrentlyInUse
def checkAgeParameters(agesDictionary):
    # Required parameters per age group
    required = ["e_escape", "a_escape", "a_to_i", "i_escape", "i_to_d",
                "i_to_h", "h_escape", "h_to_d"]
    # Track all missing parameters, so we can report all of them at once.
    missing = []
    for age, ageDict in agesDictionary.items():
        # What's missing from this age group
        missed = [param for param in required if param not in ageDict]
        if missed:
            missing.append([age, missed])
    if missing:
        for age, missed in missing:
            print(f"Age group \"{age}\" missing \"{', '.join(missed)}\"")
        raise Exception("Parameters missing")

# NotCurrentlyInUseByCoreModel
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


# CurrentlyInUse
def setUpParametersVanilla(dictOfParams):
    fromStateTrans = {}
    fromStateTrans['E'] ={'E':1-dictOfParams['e_escape'], 'A': dictOfParams['e_escape']}
    fromStateTrans['A'] = {'A':1-dictOfParams['a_escape'], 'I': dictOfParams['a_escape']*dictOfParams['a_to_i'], 'R':dictOfParams['a_escape']*(1-dictOfParams['a_to_i'])}
    fromStateTrans['I'] =  {'I':1-dictOfParams['i_escape'], 'D': dictOfParams['i_escape']*dictOfParams['i_to_d'], 'H':dictOfParams['i_escape']*(dictOfParams['i_to_h']),
                             'R':dictOfParams['i_escape']*(1-dictOfParams['i_to_h'] -dictOfParams['i_to_d']) }
    fromStateTrans['H'] = {'H':1-dictOfParams['h_escape'], 'D': dictOfParams['h_escape']*dictOfParams['h_to_d'], 'R':dictOfParams['h_escape']*(1- dictOfParams['h_to_d'])}
    fromStateTrans['R'] = {'R':1.0}
    fromStateTrans['D'] = {'D':1.0}
    return fromStateTrans


# CurrentlyInUse
def setUpParametersAges(dictByAge):
    ageToStateTrans = {}
    for age in dictByAge:
        ageToStateTrans[age] = {}
        ageToStateTrans[age] = setUpParametersVanilla(dictByAge[age])
    return ageToStateTrans
    

# CurrentlyInUse
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
        dictOfStates[currentTime+1][fella] = 'E'
    
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

# CurrentlyInUse
def countInfectionsAgeStructured(dictOfStates, time):
    total = 0
    for node in dictOfStates[time]:
        for (age, state) in dictOfStates[time][node]:
            if state == 'A' or state == 'I':
                total = total + dictOfStates[time][node][(age, state)]
    return total

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

# CurrentlyInUse
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
           
    print(dictOfStringsByNodeAndState)
    
    for node in dictOfStringsByNodeAndState:
        for state in dictOfStringsByNodeAndState[node]:
            localList = dictOfStringsByNodeAndState[node][state]
            localString = ""
            for elem in localList:
                localString = localString + "," + str(elem)
            reportString = reportString+"\n" + str(node) + "," + str(state) + localString
    return reportString

# CurrentlyInUse
# amending this so that file I/O happens outside it 
def basicSimulationInternalAgeStructure(graph, numInfected, timeHorizon, genericInfection, ageInfectionMatrix, diseaseProgressionProbs, dictOfStates):
    
    print('WARNING - FUNCTION NOT PROPERLY TESTED YET - basicSimulationInternalAgeStructure')
    ages = list(ageInfectionMatrix.values())
    timeSeriesInfection = []
    
    # dictOfStates = {}
    numInside = 100
    ages = ['y', 'm', 'o']
    states = ['S', 'E', 'A', 'I', 'H', 'R', 'D']

    # for now, we choose a random node and infect numInfected mature individuals - right now they are extra individuals, not removed from the susceptible class
    infectedNode = random.choices(list(graph.nodes()), k=1)
    for vertex in infectedNode:
        dictOfStates[0][vertex][('m', 'E')] = numInfected 

    for time in range(timeHorizon):
#         make sure the next time exists, so that we can add exposed individuals to it
        nextTime = time+1
        if nextTime not in dictOfStates:
            dictOfStates[nextTime] = {}
            for node in graph.nodes():
                dictOfStates[nextTime][node] = {}
                for age in ages:
                    for state in states:
                        dictOfStates[nextTime][node][(age, state)] = 0
        
        doInternalProgressionAllNodes(dictOfStates, time, diseaseProgressionProbs)
        
        doInteralInfectionProcessAllNodes(dictOfStates, ageInfectionMatrix, ages, time)
 
        doBetweenInfectionAgeStructured(graph, dictOfStates, time, genericInfection)

        timeSeriesInfection.append(countInfectionsAgeStructured(dictOfStates, time))

        
    return timeSeriesInfection

# CurrentlyInUse
def nodeUpdate(graph, dictOfStates, time, headString):
        print('\n\n===== BEGIN update 1 at time ' + str(time) + '=========' + headString)
        for node in list(graph.nodes()):
             print('Node ' + str(node)+ " E-A-I at mature " + str(dictOfStates[time][node][('m', 'E')]) + " " +str(dictOfStates[time][node][('m', 'A')]) + " " + str(dictOfStates[time][node][('m', 'I')]))
        print('===== END update 1 at time ' + str(time) + '=========')

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
        adds = random.sample(remainingChoice, numInEach -1)
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
        adds = random.sample(remainingChoice, min(sizeGroups -1, len(remainingChoice)))
        inAGroup.append(adds)
        for item in adds:
            strongEdges.append((start, item))
            strongEdges.append((item, start))
            for item2 in adds:
                strongEdges.append((item2, item))
                strongEdges.append((item, item2))

    return strongEdges
    

# CurrentlyInUse
def generateMeanPlot(listOfPlots):
    meanForPlot = []
    # print(listOfPlots)
    for i in range(len(listOfPlots[0])):
        sumTot = 0
        for j in range(len(listOfPlots)):
            sumTot = sumTot + listOfPlots[j][i]
        meanForPlot.append(float(sumTot)/len(listOfPlots))
    return meanForPlot
        

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


# CurrentlyInUse
# To bring this in line with the within-node infection updates (and fix a few bugs), I'm going to rework
# it so that we calculate an *expected number* of infectious contacts more directly. Then we'll distribute and
# overlap them using the same infrastructure code that we'll use for the internal version, when we add that
# Reminder: I expect the weighted edges to be the number of *expected infectious* contacts (if the giver is infectious)
#  We may need to multiply movement numbers by a probability of infection to achieve this.   
def doBetweenInfectionAgeStructured(graph, dictOfStates, currentTime, genericInfectionProb):
    totalIncomingInfectionsByNode = {}
    for receivingVertex in dictOfStates[currentTime]:
        totalSusceptHere = getTotalSuscept(dictOfStates[currentTime][receivingVertex])
        totalIncomingInfectionsByNode[receivingVertex] = 0
        if totalSusceptHere >0:
            neighbours = list(graph.predecessors(receivingVertex))
            for givingVertex in neighbours:
                    if givingVertex == receivingVertex:
                        continue
                    totalInfectedGiving = getTotalInfected(dictOfStates[currentTime][givingVertex])
                    if totalInfectedGiving >0:
                        weight = 1.0
                        if 'weight' not in graph[givingVertex][receivingVertex]:
                            print("ERROR: No weight available for edge " + str(givingVertex) + "," + str(receivingVertex) + " assigning weight 1.0")
                        else:
                            weight = graph[givingVertex][receivingVertex]['weight']
                        
                        fractionGivingInfected = totalInfectedGiving/totalIndividuals(dictOfStates[currentTime][givingVertex])
                        fractionReceivingSus = totalSusceptHere/totalIndividuals(dictOfStates[currentTime][receivingVertex])
                        totalIncomingInfectionsByNode[receivingVertex] = totalIncomingInfectionsByNode[receivingVertex] + weight*fractionGivingInfected*fractionReceivingSus

                        
#   This might over-infect - we will need to adapt for multiple infections on a single individual if we have high infection threat.  TODO raise an issue                      
    for vertex in totalIncomingInfectionsByNode:
        totalDelta = totalIncomingInfectionsByNode[vertex]
        deltaByAge = distributeInfections(dictOfStates[currentTime][vertex], totalDelta)
        for age in deltaByAge:
           dictOfStates[currentTime+1][vertex][(age, 'S')] = dictOfStates[currentTime+1][vertex][(age, 'S')] - deltaByAge[age]
           dictOfStates[currentTime+1][vertex][(age, 'E')] = dictOfStates[currentTime+1][vertex][(age, 'E')] + deltaByAge[age]



# CurrentlyInUse
#  (JE, 10 May 2020) I'm realigning this to be more using with a POLYMOD-style matrix (inlcuding the within-lockdown COMIX matrix)
# I expect the matrix entry at [age1][age2] to be the expected number of contacts in a day between age1 and age2
#  *that would infect if only one end of the contact were infectious*
#  that is, if a usual POLYMOD entry tells us that each individual of age1 is expected to have 1.2 contacts in category age2,
#  and the probability of each of these being infectious is 0.25, then I would expect the matrix going into this
# function as  ageMixingInfectionMatrix to have 0.3 in the entry [age1][age2]
def doInternalInfectionProcess(currentInternalStateDict, ageMixingInfectionMatrix, ages, time):
    newInfectedsByAge = {}
    for age in ages:
        newInfectedsByAge[age] = 0

        numSuscept = currentInternalStateDict[(age, 'S')]
        if numSuscept>0:
            numInfectiousContactsFromAges = {}
            totalNewInfectionContacts = 0
            for ageInf in ages:
                totalInfectious = currentInternalStateDict[(ageInf, 'I')] + currentInternalStateDict[(ageInf, 'A')]
                numInfectiousContactsFromAges[ageInf] = totalInfectious*ageMixingInfectionMatrix[ageInf][age]
                totalNewInfectionContacts = totalNewInfectionContacts + numInfectiousContactsFromAges[ageInf]
#      Now, given that we expect totalNewInfectionContacts infectious contacts into our age category, how much overlap do we expect?
#       and how many are with susceptible individuals? 
            totalInAge = getTotalInAge(currentInternalStateDict, age)
#       Now when we draw totalNewInfectionContacts from totalInAge with replacement, how many do we expect?
#       For now, a simplifying assumption that there are *many more* individuals in totalInAge than there are   totalNewInfectionContacts
#       So we don't have to deal with multiple infections for the same individual.  TODO - address in future code update, raise issue for this
            numNewInfected = totalNewInfectionContacts*(numSuscept/totalInAge)
            newInfectedsByAge[age] = numNewInfected
    return newInfectedsByAge


# CurrentlyInUse        
def doInteralInfectionProcessAllNodes(dictOfStates, ageMixingInfectionMatrix, ages, time):
    nextTime = time+1
    for node in dictOfStates[time]:
            newInfected = doInternalInfectionProcess(dictOfStates[time][node], ageMixingInfectionMatrix, ages, time)
            for age in newInfected:
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
        

# CurrentlyInUse
# now amending this to use data read from file on internal populations.
# the parameter dictOfPopulations  should be like the one returned by readPopulationAgeStructured
# need to add error-checking here for graceful behaviour when missing population info for a node
def setupInternalPopulations(graph, listOfStates, ages, dictOfPopulations):
    dictOfInternalStates = {}
    dictOfInternalStates[0] = {}
    currentTime = 0
    for node in list(graph.nodes()):
        dictOfInternalStates[0][node] = {}
        for state in listOfStates:
            for age in ages:
                dictOfInternalStates[0][node][(age, state)] = 0
        for age in ages:
            dictOfInternalStates[0][node][(age, 'S')] = dictOfPopulations[node]["All_Sex"][age]
        
    
    return dictOfInternalStates    
    