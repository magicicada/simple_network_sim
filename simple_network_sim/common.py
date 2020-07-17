"""
Assortment of useful functions
"""
# pylint: disable=import-error
import logging
from typing import Callable, Any, NamedTuple

import git

logger = logging.getLogger(__name__)

DEFAULT_GITHUB_REPO = "https://github.com/ScottishCovidResponse/simple_network_sim.git"


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
        for plot in listOfPlots:
            sumTot = sumTot + plot[i]
        meanForPlot.append(float(sumTot) / len(listOfPlots))
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


class RepoInfo(NamedTuple):
    """
    The info needed by the data pipeline API
    """
    git_sha: str
    uri: str
    is_dirty: bool


def get_repo_info() -> RepoInfo:
    """
    Retrieves the current git sha and uri for the current git repo
    :return: A RepoInfo object. If not inside a git repo, is_dirty will be True, git_sha empty and uri will be a
             default value
    """
    try:
        repo = git.Repo()
    except git.InvalidGitRepositoryError:
        return RepoInfo(git_sha="", uri=DEFAULT_GITHUB_REPO, is_dirty=True)
    else:
        return RepoInfo(
            git_sha=repo.head.commit.hexsha,
            uri=next(repo.remote("origin").urls),
            is_dirty=repo.is_dirty(),
        )
