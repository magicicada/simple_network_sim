import copy


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
# amending this so that file I/O happens outside it 
def basicSimulationInternalAgeStructure(rand, graph, numInfected, timeHorizon, ageInfectionMatrix, diseaseProgressionProbs, dictOfStates):
    timeSeriesInfection = []

    # for now, we choose a random node and infect numInfected mature individuals
    infectedNode = rand.choices(list(graph.nodes()), k=1)
    for vertex in infectedNode:
        dictOfStates[0][vertex][('m', 'E')] = numInfected
        assert dictOfStates[0][vertex][('m', 'S')] >= numInfected, "cannot infect more than number of susceptible"
        dictOfStates[0][vertex][('m', 'S')] -= numInfected

    for time in range(timeHorizon):
        # make sure the next time exists, so that we can add exposed individuals to it
        nextTime = time + 1
        if nextTime not in dictOfStates:
            dictOfStates[nextTime] = copy.deepcopy(dictOfStates[time])
            for region in dictOfStates[nextTime].values():
                for state in region.keys():
                    region[state] = 0.0

        doInternalProgressionAllNodes(dictOfStates, time, diseaseProgressionProbs)

        doInteralInfectionProcessAllNodes(dictOfStates, ageInfectionMatrix, time)

        doBetweenInfectionAgeStructured(graph, dictOfStates, time)

        timeSeriesInfection.append(countInfectionsAgeStructured(dictOfStates, time))

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
