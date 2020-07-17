import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


# CurrentlyInUse
def generateMeanPlot(listOfPlots):
    """From a list of disease evolution timeseries, compute the average evolution.

    :param listOfPlots: List of disease evolutions
    :type listOfPlots: list
    :return: The average evolution
    :rtype: list
    """
    meanForPlot = []
    logger.debug(listOfPlots)
    for i in range(len(listOfPlots[0])):
        sumTot = 0
        for j in range(len(listOfPlots)):
            sumTot = sumTot + listOfPlots[j][i]
        meanForPlot.append(float(sumTot)/len(listOfPlots))
    return meanForPlot


class Lazy:
    """
    This class allows lazy evaluation of logging expressions. The idiom to accomplish that can be better explained in
    the example below::

        logger.info("The value of z is: %s", lazy(lambda: x + y))

    that will cause ``x + y`` to only be evaluated if the log level is info.

    :param f: A function which takes no parameters and which will only be evaluated when str is called in the returning
              object
    """
    def __init__(self, f: Callable[[], Any]):
        self.f = f

    def __str__(self):
        return str(self.f())

    def __repr__(self):
        return repr(self.f())
