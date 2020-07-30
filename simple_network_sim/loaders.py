"""This module contains functions and classes to read and check input files."""

import datetime
import json
import math
from typing import Any, Dict, NamedTuple, List, Tuple, Union

import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

# Type aliases used to make the types for the functions below easier to read
Age = str
Compartment = str
NodeName = str


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


def readCompartmentRatesByAge(table: pd.DataFrame,) -> Dict[Age, Dict[Compartment, Dict[Compartment, float]]]:
    """
    Read a file containing a list of age-specific epidemiological parameters.

    Epidemiological parameters are transition rates between or out of epidemiological compartments,
    here they differ by age group.

    :param table: Age transition data
    :return: nested dictionary with progression rates
    """
    agesDictionary: Dict[Age, Dict[Compartment, Dict[Compartment, float]]] = {}
    for row in table.to_dict(orient="row"):
        compartments = agesDictionary.setdefault(row["age"], {})
        transitions = compartments.setdefault(row["src"], {})
        transitions[row["dst"]] = float(row["rate"])
    return _checkAgeParameters(agesDictionary)


def readPopulationAgeStructured(table: pd.DataFrame) -> Dict[NodeName, Dict[Age, int]]:
    """Read a file containing population data.

    Population is labelled by node ID, sex and age. Sex is currently ignored.

    :param table: Population data
    :return: Nested dict with age-stratified population in each node
    """
    dictOfPops: Dict[NodeName, Dict[Age, int]] = {}

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


# making this general to include arbitrary future attributes.  Location is the primary one for right now
# keeps them in a dictionary and returns that.  Keys are
def readNodeAttributesJSON(filename):
    """
    Read a file containing node (health board) attributes.

    This might include health board name and geographic co-ordinates.

    :param filename: Health board attributes in json format.
    :type filename: file-like object
    :return: Dictionary of health board attributes.
    """
    f = open(filename,)
    node_data = json.load(f)
    return node_data


# at the moment this uses vanilla networkx edge list reading - needs weights
#  I've set it apart as its own function in case we want to do anything fancier with edge files
# in future - e.g. sampling, generating movements, whatever
# it should return a networkx graph, ideally with weighted edges
# eventual replacement with HDF5 reading code?
def genGraphFromContactFile(commutes: pd.DataFrame) -> nx.DiGraph:
    """Read a file containing edge weights between nodes.

    Pairs of nodes are listed in the file by source, destination, weight and adjustment.

    :param commutes: Weighted edge list.
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
    """
    Factors used to dampen or heighten the movement of people between nodes and contact of people within nodes
    """
    movement: float
    contact: float


def _assertPositiveNumber(value: float):
    if value < 0.0 or math.isinf(value) or math.isnan(value):
        raise ValueError(f"{value} must be a positive number")


def readMovementMultipliers(table: pd.DataFrame) -> Dict[datetime.date, Multiplier]:
    """Read file containing movement multipliers by time.

    :param table: pandas DataFrame containing movement multipliers
    :return: A dict of ints (time) pointing to a (named)tuple (contact=float, movement=float). Contact will alter how
             the disease spreads inside of a node, whereas movement will change how it spread across nodes
    """
    multipliers = {}
    for row in table.to_dict(orient="row"):
        if not isinstance(row["Date"], str):
            raise ValueError("Date must be string")

        date = datetime.datetime.strptime(row["Date"], '%Y-%m-%d').date()

        movement = float(row["Movement_Multiplier"])
        _assertPositiveNumber(movement)

        contact = float(row["Contact_Multiplier"])
        _assertPositiveNumber(contact)

        multipliers[date] = Multiplier(movement=movement, contact=contact)

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


def readABCSMCParameters(parameters: pd.DataFrame) -> Dict:
    """
    Transforms the API output of parameters into the internal representation: a dict
    :param parameters: pandas DataFrame with the raw data from the API
    :return: a dict of inference parameters
    """
    if parameters.size == 0:
        raise ValueError("Parameters cannot be empty")

    if "Parameter" not in parameters.columns:
        raise ValueError("'Parameter' column should be in ABCSMC parameters")

    if "Value" not in parameters.columns:
        raise ValueError("'Value' column should be in ABCSMC parameters")

    parameters = parameters.set_index("Parameter").Value.to_dict()

    parameters["n_smc_steps"] = int(parameters["n_smc_steps"])
    parameters["n_particles"] = int(parameters["n_particles"])
    parameters["infection_probability_shape"] = float(parameters["infection_probability_shape"])
    parameters["infection_probability_kernel_sigma"] = float(parameters["infection_probability_kernel_sigma"])
    parameters["initial_infections_stddev"] = float(parameters["initial_infections_stddev"])
    parameters["initial_infections_stddev_min"] = float(parameters["initial_infections_stddev_min"])
    parameters["initial_infections_kernel_sigma"] = float(parameters["initial_infections_kernel_sigma"])
    parameters["contact_multipliers_stddev"] = float(parameters["contact_multipliers_stddev"])
    parameters["contact_multipliers_kernel_sigma"] = float(parameters["contact_multipliers_kernel_sigma"])

    partitions = [datetime.datetime.strptime(d, '%Y-%m-%d').date()
                  for d in parameters["contact_multipliers_partitions"].split(", ")]
    partitions.append(datetime.date.max)
    partitions.insert(0, datetime.date.min)
    parameters["contact_multipliers_partitions"] = partitions

    return parameters


def readHistoricalDeaths(historical_deaths: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the API output of target into the internal representation: a pd.DataFrame
    :param historical_deaths: pandas DataFrame with the raw data from the API
    :return: a pd.DataFrame of historical deaths by HB
    """
    if historical_deaths.size == 0:
        raise ValueError("With an empty target no inference can take place")

    historical_deaths = historical_deaths.set_index("Week beginning")

    if np.any(historical_deaths.values < 0):
        raise ValueError("Cannot have negative deaths")

    historical_deaths.index = pd.to_datetime(historical_deaths.index)

    return historical_deaths


def readInfectionProbability(df: pd.DataFrame) -> Dict[datetime.date, float]:
    """
    Transforms the dataframe from the data API into a dict usable inside the model

    :param df: a timeseries of infection probabilities
    :return: a timeseries of infection probabilities in the dict format
    """
    if df.empty:
        raise ValueError("Dataframe must be non empty")

    probs: Dict[datetime.date, float] = {}
    for row in df.to_dict(orient="row"):
        if not isinstance(row["Date"], str):
            raise ValueError("Date must be string")

        date = datetime.datetime.strptime(row["Date"], '%Y-%m-%d').date()
        value = float(row["Value"])
        if value < 0.0 or value > 1.0 or math.isnan(value):
            raise ValueError("infection probability must be between 0 and 1")
        probs[date] = value

    return probs


def readRandomSeed(df: pd.DataFrame) -> int:
    """
    Transforms the dataframe from the data API into a bool usable inside the model

    :param df: a dataframe containing the random seed
    :return: a value of using the random seed as an int
    """
    if df is None:
        return 0

    assert len(df) == 1
    assert df.columns == ["Value"]

    for row in df.to_dict(orient="row"):
        if isinstance(row["Value"], str):
            seed = int(row["Value"])
        else:
            seed = row["Value"]

        if not isinstance(seed, int):
            raise ValueError("Seed must be an int")

        if seed < 0:
            raise ValueError("Seed must be positive")

        return seed

    raise ValueError("No seed found")


def readTrials(df: pd.DataFrame) -> int:
    """
    Transforms the dataframe from the data API into a bool int inside the model

    :param df: The dataframe containing the number of trials
    :return: the number of trials to run
    """
    assert len(df) == 1
    assert df.columns == ["Value"]

    for row in df.to_dict(orient="row"):
        trials = row["Value"]

        if not isinstance(trials, int):
            raise ValueError("trials must be an int")

        if trials < 1:
            raise ValueError("trials must be > 0")

        return trials

    raise ValueError("Dataframe must have at least one row")


def readStartEndDate(df: pd.DataFrame) -> Tuple[datetime.date, datetime.date]:
    """
    Transforms the dataframe from the data API into a bool int inside the model
    :param df: The dataframe containing the starting date of the model
    :return: The starting date of the model
    """
    if len(df) != 2:
        raise ValueError("DataFrame must be of size 2")
    if not all(df.columns == ["Parameter", "Value"]):
        raise ValueError("There must be 2 columns names 'Parameter' and 'Value'")
    if ("start_date" not in list(df.Parameter)) or ("end_date" not in list(df.Parameter)):
        raise ValueError("Both a start and end date must be provided")

    start_date = datetime.datetime.strptime(df.at[0, "Value"], '%Y-%m-%d').date()
    end_date = datetime.datetime.strptime(df.at[1, "Value"], '%Y-%m-%d').date()

    return start_date, end_date


def readStochasticMode(df: pd.DataFrame) -> bool:
    """
    Transforms the dataframe from the data API into a bool usable inside the model

    :param df: a dataframe with the stochastic mode
    :return: a value of using the stochastic mode as a bool
    """
    if df is None:
        return False

    assert len(df) == 1
    assert df.columns == ["Value"]

    for row in df.to_dict(orient="row"):
        stochastic_mode = row["Value"]

        if not isinstance(stochastic_mode, bool):
            raise ValueError("stochastic_mode must be bool")

        return bool(stochastic_mode)

    raise ValueError("Dataframe must contain at least one row")


class AgeRange:
    """A helper class for an age range.
    
    The age_group parameter can be any string, but it is usually in the format [a,b) or 70+
    """

    def __init__(self, age_group: str):
        """Initialise."""
        self.age_group = age_group

    def __str__(self) -> str:
        """Return a string representation of the current AgeRange instance."""
        return self.age_group

    def __eq__(self, other: Any) -> bool:
        """Return true if "other" is the same as the current AgeRange instance."""
        if not isinstance(other, AgeRange):
            return False
        return self.age_group == other.age_group

    def __neq__(self, other: Any) -> bool:  # type: ignore
        """Return true if "other" is not the same as the current AgeRange instance."""
        if not isinstance(other, AgeRange):
            return False
        return not self == other

    def __hash__(self) -> int:
        """Return a hash of the current AgeRange instance."""
        return hash(self.age_group)


class MixingRow:
    """
    One row of a mixing table.

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
    """
    Stores expected numbers of interactions between people of different ages.

    Stores expected number of interactions per day a given person will
    have with some other person, based on the ages of the two people, or given
    age-ranges.

    Examples:

    .. code-block:: python

        mm = MixingMatrix(api.read_table("human/mixing-matrix"))
        print(mm[28][57])  # Prints the expected number of interactions a 28 year old

    would have with a 57 year old in a day

    .. code-block:: python

        print(mm["[30,40)"]["70+"]); print(mm[(30,40)]["70+"])

    prints the expected number of interactions someone in the age range [30-40) would have with someone aged 70 or older
    in any given day.

    :param mixing_table: Raw DataFrame from the data API. The expected columns are: source, target and mixing (value).
    """

    def __init__(self, mixing_table: Union[pd.DataFrame, Dict[str, Dict[str, float]]]):
        """Initialise."""

        if isinstance(mixing_table, pd.DataFrame):
            self._matrix = {
                group_name: MixingRow(list(group["target"]), list(group["mixing"]))
                for group_name, group in mixing_table.groupby("source")
            }
        elif isinstance(mixing_table, dict):
            self._matrix = mixing_table
        else:
            raise ValueError("Wrong data type passed in constructor of MixingMatrix")

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
