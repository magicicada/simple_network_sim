import random
import json

dictOfNodes = {}
for node in range(100):
   dictOfNodes[node] = {}
   dictOfNodes[node]['name'] = str(node)
   dictOfNodes[node]['xLoc'] = str(random.random())
   dictOfNodes[node]['yLoc'] = str(random.random())

with open('sampleNodeLocations.json', 'w') as fp:
    json.dump(dictOfNodes, fp, indent=4)
