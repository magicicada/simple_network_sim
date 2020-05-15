import networkx as nx
import random
import sys
from collections import Counter
import matplotlib.pyplot as plt
import json

from . import common, network_of_populations as ss, loaders

    
 #  A bit of sample model operation.     

compNames = ['S', 'E', 'A', 'I', 'H', 'R', 'D']

baseWeight = 0.5
# strongWeight = 0.8
ages=['y', 'm', 'o']
numInfected = 10
genericInfection = 0.1
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

baseGraph = nx.path_graph(10)
for (u, v) in list(baseGraph.edges()):
     baseGraph[u][v]['weight'] = baseWeight

params = loaders.readParametersAgeStructured(sys.argv[1])

ageToTrans = ss.setUpParametersAges(params)


dictOfPops = loaders.readPopulationAgeStructured(sys.argv[2])

graph = loaders.genGraphFromContactFile(sys.argv[3])

states = ss.setupInternalPopulations(graph, compNames, ages, dictOfPops)

basicPlots = []

time = 200
numTrials = 100

for i in range(numTrials):
     basicPlots.append(ss.basicSimulationInternalAgeStructure(random.Random(), graph, numInfected, time, genericInfection, ageInfectionMatrix, ageToTrans, states))

plt.plot(common.generateMeanPlot(basicPlots), color ='dodgerblue', label='basic')

plt.savefig(sys.argv[4])

# 
# 
# debuggingState ={('y', 'S'): 100, ('m', 'S'): 100, ('o', 'S'): 100, ('y', 'E'): 100, ('m', 'E'): 100, ('o', 'E'): 100, ('y', 'A'): 100, ('m', 'A'): 100,
#     ('o', 'A'): 100, ('y', 'I'): 100, ('m', 'I'): 100, ('o', 'I'): 100, ('y', 'H'): 100, ('m', 'H'): 100, ('o', 'H'): 100,
#     ('y', 'R'): 100, ('m', 'R'): 100, ('o', 'R'): 100, ('y', 'D'): 100, ('m', 'D'): 100, ('o', 'D'): 100}
# 
# 
# outState = ss.internalStateDiseaseUpdate(debuggingState, ageToTrans)
# print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
# for key in outState:
#     print(str(key) + str(outState[key]))




