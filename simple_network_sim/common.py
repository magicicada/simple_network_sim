import logging

logger = logging.getLogger(__name__)

# CurrentlyInUse
def generateMeanPlot(listOfPlots):
    meanForPlot = []
    logger.debug(listOfPlots)
    for i in range(len(listOfPlots[0])):
        sumTot = 0
        for j in range(len(listOfPlots)):
            sumTot = sumTot + listOfPlots[j][i]
        meanForPlot.append(float(sumTot)/len(listOfPlots))
    return meanForPlot
