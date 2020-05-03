import networkx as nx
import random
import sys
from collections import Counter
import matplotlib.pyplot as plt
import json
import simplestNetworkSim_covidAdapted as ss
    
 #  A bit of sample model operation.     

compNames = ['S', 'E', 'A', 'I', 'H', 'R', 'D']
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
states = ss.setupInternalPopulations(baseGraph, compNames, ages)
# for node in states[0]:
#     print(states[0][node])

params = ss.readParametersAgeStructured(sys.argv[1])
for guy in params:
    print (params[guy])

ageToTrans = ss.setUpParametersAges(params)
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
time = 20
numTrials = 1
# for i in range(numTrials): 
#     withGroups.append(basicSimulation(baseGraph, 4, time, 0.1))
# print('Done withGroups')
for i in range(numTrials):
     basicPlots.append(ss.basicSimulationInternalAgeStructure(baseGraph, numInfected, time, genericInfection, ageInfectionMatrix, ageToTrans))
# # print('Done basic')
# addIllicitEdges(baseGraph, sizeGroups^2*numGroups)
# for i in range(numTrials):
#     withIllicit.append(basicSimulation(baseGraph, 4, time, 0.1))
# print('Done withIllicit')
# 
# plt.plot(generateMeanPlot(withIllicit), color='maroon', label = 'with_illicit')
# plt.plot(generateMeanPlot(withGroups), color='green', label = 'with_groups')
plt.plot(ss.generateMeanPlot(basicPlots), color = 'dodgerblue', label='basic')
# plt.legend()
# # plt.ylim(top=200)
# 
plt.savefig('withAges.pdf')
