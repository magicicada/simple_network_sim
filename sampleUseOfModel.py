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

# 
# baseGraph = generateHouseholdsAggregateGraph(10, 0.06)
#
baseGraph = nx.path_graph(10)
for (u, v) in list(baseGraph.edges()):
     baseGraph[u][v]['weight'] = baseWeight
#     

# for node in states[0]:
#     print(states[0][node])

params = ss.readParametersAgeStructured(sys.argv[1])
# for guy in params:
#     print (params[guy])

ageToTrans = ss.setUpParametersAges(params)
# for age in ageToTrans:
#     print (age)
#     print( ageToTrans[age])


dictOfPops = ss.readPopulationAgeStructured(sys.argv[2])
# for guy in dictOfPops:
#     print("\n" + guy)
#     print (dictOfPops[guy])
#     
# print("\n\n\n\n")
graph = ss.genGraphFromContactFile(sys.argv[3])
# # print (graph.edges())
# # print (graph['S08000015']['S08000016'])


# print("\n\n\n\n states examination")
states = ss.setupInternalPopulations(graph, compNames, ages, ss.readPopulationAgeStructured(sys.argv[2]))
print(states.keys())
for node in states[0]:
    print("\n\n" + str(node))
    print (states[0][node])





# # pos = nx.get_node_attributes(baseGraph, 'pos')
# # strongEdges = generateChildcareEdgesAggregate(baseGraph, numGroups, sizeGroups)
# # for (u, v) in strongEdges:
# #     baseGraph.add_edge(u, v)
# #     baseGraph[u][v]['weight'] = strongWeight
# # 
# # baseGraph.add_edges_from(strongEdges)
# # 
# # longDistance = generateIllicitEdges(baseGraph, 100)
# # 
# # paraDict = readParameters(sys.argv[1])
# # fromStateTrans = setUpParametersVanilla(paraDict)
# # print(fromStateTrans)
# # 
basicPlots = []
# # withGroups = []
# # withIllicit = []
time = 200
numTrials = 1
# # for i in range(numTrials): 
# #     withGroups.append(basicSimulation(baseGraph, 4, time, 0.1))
# # print('Done withGroups')

print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
for i in range(numTrials):
     basicPlots.append(ss.basicSimulationInternalAgeStructure(graph, numInfected, time, genericInfection, ageInfectionMatrix, ageToTrans, states))
# # # print('Done basic')
# # addIllicitEdges(baseGraph, sizeGroups^2*numGroups)
# # for i in range(numTrials):
# #     withIllicit.append(basicSimulation(baseGraph, 4, time, 0.1))
# # print('Done withIllicit')
# # 
# # plt.plot(generateMeanPlot(withIllicit), color='maroon', label = 'with_illicit')
# # plt.plot(generateMeanPlot(withGroups), color='green', label = 'with_groups')
plt.plot(ss.generateMeanPlot(basicPlots), color = 'dodgerblue', label='basic')
# # plt.legend()
# # # plt.ylim(top=200)
# # 
plt.savefig('withAges.pdf')

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




