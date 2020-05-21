import copy
import random
import sys
import matplotlib.pyplot as plt

from . import common, network_of_populations as ss

numInfected = 10

network = ss.createNetworkOfPopulation(sys.argv[1], sys.argv[2], sys.argv[3])

basicPlots = []

time = 200
numTrials = 100

initialState = copy.deepcopy(network.states[0])
for region in random.choices(list(network.graph.nodes()), k=numTrials):
    network.states[0] = initialState
    ageDistribution = {"m": 1.0}
    ss.exposeRegions([region], numInfected, ageDistribution, network.states[0])
    basicPlots.append(ss.basicSimulationInternalAgeStructure(network, time))

plt.plot(common.generateMeanPlot(basicPlots), color ='dodgerblue', label='basic')

plt.savefig(sys.argv[4])
