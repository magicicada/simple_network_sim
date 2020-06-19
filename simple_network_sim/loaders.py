"""This module contains functions and classes to read and check input files."""

import csv
import json
import math
import re
from typing import Dict, TextIO, NamedTuple, List

import networkx as nx
import pandas as pd

# Type aliases used to make the types for the functions below easier to read
Age = str
Compartment = str
NodeName = str


# CurrentlyInUse
def _checkAgeParameters(agesDictionary):
    """Check the consistency of data within the ages dictionary.

    :param agesDictionary:
    :type agesDictionary: dictionary
    :return: agesDictionary
    """
    all_compartments = None
    for age, compartments in agesDictionary.items():
        if all_compartments is None:
            all_compartments = list(compartments.keys())
        else:
            assert all_compartments == list(compartments.keys()), f"compartments mismatch in {age}"
        for compartment, transitions in compartments.items():
            assert compartment in transitions.keys(), f"{age},{compartment} does not have self referencing key"
            assert math.isclose(sum(transitions.values()), 1.0), f"{age},{compartment} transitions do not add up to 1.0"
            for new_name, prob in transitions.items():
                assert 0.0 <= prob <= 1.0, f"{age},{compartment},{new_name},{prob} not a valid probability"

    return agesDictionary


# CurrentlyInUse
def readCompartmentRatesByAge(table: pd.DataFrame,) -> Dict[Age, Dict[Compartment, Dict[Compartment, float]]]:
    """Read a file containing a list of age-specific epidemiological parameters.
    
    Epidemiological parameters are transition rates between or out of epidemiological compartments,
    here they differ by age group.

    :param table: Age transition data
    :return: nested dictionary with progression rates
    """
    agesDictionary = {}
    for row in table.to_dict(orient="row"):
        compartments = agesDictionary.setdefault(row["age"], {})
        transitions = compartments.setdefault(row["src"], {})
        transitions[row["dst"]] = float(row["rate"])
    return _checkAgeParameters(agesDictionary)


# CurrentlyInUse
def readPopulationAgeStructured(table: pd.DataFrame) -> Dict[NodeName, Dict[Age, int]]:
    """Read a file containing population data.

    Population is labelled by node ID, sex and age. Sex is currently ignored.
    
    :param fp: Population data
    :return: Nested dict with age-stratified population in each node
    """
    dictOfPops = {}

    for row in table.to_dict(orient="row"):
        board = dictOfPops.setdefault(row["Health_Board"], {})
        total = int(row["Total"])
        if total < 0:
            raise ValueError(f"invalid total {total}")
        # We are ignoring the Sex column here. The same age group will appear multiple times (once per age) and we will
        # end up just grouping them all together
        board.setdefault(row["Age"], 0)
        board[row["Age"]] += total

    return dictOfPops


# CurrentlyInUse
# making this general to include arbitrary future attributes.  Location is the primary one for right now
# keeps them in a dictionary and returns that.  Keys are
def readNodeAttributesJSON(filename):
    """Read a file containing node (health board) attributes.

    This might include health board name and geographic co-ordinates.

    :param filename: Health board attributes in json format. 
    :type filename: file-like object
    :return: Dictionary of health board attributes.
    """
    f = open(filename,)
    node_data = json.load(f)
    return node_data


# CurrentlyInUse
# at the moment this uses vanilla networkx edge list reading - needs weights
#  I've set it apart as its own function in case we want to do anything fancier with edge files
# in future - e.g. sampling, generating movements, whatever
# it should return a networkx graph, ideally with weighted edges
# eventual replacement with HDF5 reading code?
def genGraphFromContactFile(commutes: pd.DataFrame) -> nx.DiGraph:
    """Read a file containing edge weights between nodes.

    Pairs of nodes are listed in the file by source, destination, weight and adjustment.

    :param commutes: Weighted edge list.
    :type pd.DataFrame: pandas DataFrame.
    :return: `networkx.classes.digraph.DiGraph` object representing the graph.
    """
    G = nx.convert_matrix.from_pandas_edgelist(commutes, edge_attr=True, create_using=nx.DiGraph)
    for edge in G.edges.data():
        assert edge[2]["weight"] >= 0.0
        assert edge[2]["delta_adjustment"] >= 0.0
    return G


def readInitialInfections(df: pd.DataFrame) -> Dict[NodeName, Dict[Age, float]]:
    """Read initial numbers of infected individuals by health board and age.

    :param df: raw data to be loaded
    :return: nested dict with number of infected per age per node
    """
    infections: Dict[str, Dict[str, float]] = {}
    for row in df.to_dict(orient="row"):
        if float(row["Infected"]) >= 0.0 and row["Infected"] != math.inf:
            infections.setdefault(row["Health_Board"], {})[row["Age"]] = row["Infected"]
        else:
            raise ValueError(f"Invalid infected value: {row['Infected']}")
    return infections


class Multiplier(NamedTuple):
    movement: float
    contact: float


def _assertPositiveNumber(value: float):
    if value < 0.0 or math.isinf(value) or math.isnan(value):
        raise ValueError(f"{value} must be a positive number")


def readMovementMultipliers(table: pd.DataFrame) -> Dict[int, Multiplier]:
    """Read file containing movement multipliers by time.

    :param table: pandas DataFrame containing movement multipliers
    :return: A dict of ints (time) pointing to a (named)tuple (contact=float, movement=float). Contact will alter how
             the disease spreads inside of a node, whereas movement will change how it spread across nodes
    """
    multipliers = {}
    for row in table.to_dict(orient="row"):
        time = int(row["Time"])
        if time < 0:
            raise ValueError("can't have negative time")

        movement = float(row["Movement_Multiplier"])
        _assertPositiveNumber(movement)

        contact = float(row["Contact_Multiplier"])
        _assertPositiveNumber(contact)

        multipliers[time] = Multiplier(movement=movement, contact=contact)

    return multipliers


def readInfectiousStates(infectious_states: pd.DataFrame) -> List[Compartment]:
    """
    Transforms the API output of infectious_states into the internal representation: a list of strings
    :param infectious_states: pandas DataFrame with the raw data from the API
    :return: a list of strings of infectious compartments
    """
    if infectious_states.size == 0:
        return []
    return list(infectious_states.Compartment)


def readInfectionProbability(df: pd.DataFrame) -> Dict[int, float]:
    """
    Transforms the dataframe from the data API into a dict usable inside the model

    :param df: a timeseries of infection probabilities
    :return: a timeseries of infection probabilities in the dict format
    """
    probs: Dict[int, float] = {}
    has_time_zero = False
    for row in df.to_dict(orient="row"):
        time = int(row["Time"])
        if time == 0:
            has_time_zero = True
        elif time < 0:
            raise ValueError("can't have negative time")
        value = float(row["Value"])
        if value < 0.0 or value > 1.0 or math.isnan(value):
            raise ValueError("infection probabilty must be between 0 and 1")
        probs[time] = value

    if not has_time_zero:
        raise ValueError("the infection probability needs to be present since the time 0")

    return probs


def _check_overlap(one, two):
    """Check two AgeRange objects to see if they overlap.
    
    If they do, raise an Exception.
    :param one:
    :type one: simple_network_sim.loaders.AgeRange
    :param two:
    :type two: simple_network_sim.loaders.AgeRange
    """
    assert one._upper <= two._lower or two._upper <= one._lower, \
            (f"Overlap in age ranges with {one} and {two}")


class AgeRange:
    """A helper class for an age range.
    
    The age_group parameter can be any string, but it is usually in the format [a,b) or 70+
    """

    def __init__(self, age_group: str):
        """Initialise."""
        self.age_group = age_group

    def __str__(self):
        """Return a string representation of the current AgeRange instance."""
        return self.age_group

    def __eq__(self, other: "AgeRange"):
        """Return true if "other" is the same as the current AgeRange instance."""
        return self.age_group == other.age_group

    def __neq__(self, other: "AgeRange"):
        """Return true if "other" is not the same as the current AgeRange instance."""
        return not self == other

    def __hash__(self):
        """Return a hash of the current AgeRange instance."""
        return hash(self.age_group)


class MixingRow:
    """One row of a mixing table.
    
    This is a helper class. A row represents a given population, 
    and can return the expected number of interactions (per unit time) 
    a member of this population will have with some other target.
    
    :param ages: Ages
    :type ages: list
    :param interactions: Interactions / person / day
    :type interactions: list

    Both lists must be the same length.

    Upon initialization, the property `self._entries` is set to a dictionary mapping 
    AgeRange objects to numbers of interactions / person / day.
    """

    def __init__(self, ages: List[str], interactions: List[str]):
        """Initialise."""        
        self._entries = {}
        for age, interact in zip(ages, interactions):
            self._entries[age] = float(interact)

    def __getitem__(self, age: str) -> float:
        """Return expected number of interactions.
        
        Return the expected number of interactions (per day) that someone from
        this MixingRow would have with someone with the given age, or age
        range.
        """
        return self._entries[age]

    def __str__(self):
        """Return a string representing the current MixingRow instance."""
        return "[" + ", ".join(f"{str(key)}: {str(val)}"
                               for key, val in self._entries.items()) + "]"

    def __iter__(self):
        """Iterate through age-interactions dictionary."""
        return iter(age_range for age_range in self._entries)


class MixingMatrix:
    """Stores expected numbers of interactions between people of different ages.
    
    Stores expected number of interactions per day a given person will
    have with some other person, based on the ages of the two people, or given
    age-ranges.
    
    Examples:    
    `mm = MixingMatrix(api.read_table("human/mixing-matrix")`
    `print(mm[28][57])` Prints the expected number of interactions a 28 year old
    would have with a 57 year old in a day
    `print(mm["[30,40)"]["70+"])` or `print(mm[(30,40)]["70+"])` Prints the expected number of interactions someone in the
    age range [30-40) would have with someone aged 70 or older
    in any given day.

    :param mixing_table: Raw DataFrame from the data API. The expected columns are: source, target and mixing (value).
    """

    def __init__(self, mixing_table: pd.DataFrame):
        """Initialise."""
        self._matrix = {
            group_name: MixingRow([target for target in group["target"]], list(group["mixing"]))
            for group_name, group in mixing_table.groupby("source")
        }

    def __getitem__(self, age: str) -> MixingRow:
        """Return MixingRow for given age.
        
        Gets a MixingRow for the given age, which in turn can give the
        expected number of interactions. Most often, you will probably want to
        just use MixingMatrix[age1][age2] to get the expected number of
        interactions that a person of age1 will have with a person of age2.
        Note that either age1 or age2 can be numbers, or age ranges.
        """
        return self._matrix[age]

    def __str__(self):
        """Return a string representing the current MixingMatrix instance."""
        return "\n".join(f"{str(key)}: {str(row)}" for key, row in self._matrix.items())

    def __iter__(self):
        """Iterate by row.

        Iterator that iterates over the matrix keys (a key points to a row). The values returned by the iterator will
        all be strings, since that's how the public interface when indexing the matrix.
        """
        return iter(age_range for age_range in self._matrix)
