import json

import networkx as nx


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
    boardInd = 0
    sexInd = 1
    totalAgeInd = 2
    youngInd = 3
    matureInd = 4
    oldInd = 5
    with open(filename, 'r') as f:
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
# making this general to include arbitrary future attributes.  Location is the primary one for right now
# keeps them in a dictionary and returns that.  Keys are
def readNodeAttributesJSON(filename):
    f = open(filename,)
    node_data = json.load(f)
    return node_data


# CurrentlyInUse
# at the moment this uses vanilla networkx edge list reading - needs weights
#  I've set it apart as its own function in case we want to do anything fancier with edge files
# in future - e.g. sampling, generating movements, whatever
# it should return a networkx graph, ideally with weighted edges
# eventual replacement with HDF5 reading code?
def genGraphFromContactFile(filename):
    G = nx.read_edgelist(filename, create_using=nx.DiGraph, delimiter=",", data=(('weight', float),))
    return G
