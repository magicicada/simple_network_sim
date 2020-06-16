import random
import json

"""Create a JSON to represent a graph of 100 random nodes.

:param nbr_nodes: Number of nodes to create in the random graph.
:type nbr_nodes: int
:return: File with JSON representing the random graph.
:rtype: .json file
"""

nbr_nodes = 100

dictOfNodes = {}
for node in range(nbr_nodes):
    dictOfNodes[node] = {}
    dictOfNodes[node]['name'] = str(node)
    dictOfNodes[node]['xLoc'] = str(random.random())
    dictOfNodes[node]['yLoc'] = str(random.random())

with open('sampleNodeLocations.json', 'w') as fp:
    json.dump(dictOfNodes, fp, indent=4)
