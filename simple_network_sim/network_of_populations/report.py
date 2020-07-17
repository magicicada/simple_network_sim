# from simple_network_sim.network_of_populations import logger


# # NotCurrentlyInUse
# # this takes a dictionary of states at times at nodes, and returns a string
# # reporting the number of people in each state at each node at each time.
# # aggregates by age
# def basicReportingFunction(dictOfStates):
#     reportString = ""
#     dictOfStringsByNodeAndState = {}
# #     Assumption: all nodes exist at time 0
#     for node in dictOfStates[0]:
#         dictOfStringsByNodeAndState[node] = {}
#         for (age, state) in dictOfStates[0][node]:
#             dictOfStringsByNodeAndState[node][state] = []
#     for time in dictOfStates:
#         for node in dictOfStates[time]:
#             numByState = {}
#             for (age, state) in dictOfStates[time][node]:
#                 if state not in numByState:
#                     numByState[state] = 0
#                 numByState[state] = numByState[state] + dictOfStates[time][node][(age, state)]
#             for state in numByState:
#                 dictOfStringsByNodeAndState[node][state].append(numByState[state])

#     logger.debug(dictOfStringsByNodeAndState)

#     for node in dictOfStringsByNodeAndState:
#         for state in dictOfStringsByNodeAndState[node]:
#             localList = dictOfStringsByNodeAndState[node][state]
#             localString = ""
#             for elem in localList:
#                 localString = localString + "," + str(elem)
#             reportString = reportString+"\n" + str(node) + "," + str(state) + localString
#     return reportString


# # NotCurrentlyInUse
# def nodeUpdate(graph, dictOfStates, time, headString):
#     print('\n\n===== BEGIN update 1 at time ' + str(time) + '=========' + headString)
#     for node in list(graph.nodes()):
#         print('Node ' + str(node) + " E-A-I at mature " + str(dictOfStates[time][node][('m', 'E')]) + " " +
#               str(dictOfStates[time][node][('m', 'A')]) + " " + str(dictOfStates[time][node][('m', 'I')]))
#     print('===== END update 1 at time ' + str(time) + '=========')
