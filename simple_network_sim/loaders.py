import csv
import json
import re

import networkx as nx


# CurrentlyInUse
def checkAgeParameters(agesDictionary):
    # Required parameters per age group
    # TODO(rafael): if we decide to take the compartments graph as an input, we will need to revisit this list
    required = ["e_escape", "a_escape", "a_to_i", "i_escape", "i_to_d",
                "i_to_h", "h_escape", "h_to_d"]
    # Track all missing parameters, so we can report all of them at once.
    missing = []
    for age, ageDict in agesDictionary.items():
        # What's missing from this age group
        missed = [param for param in required if param not in ageDict]
        if missed:
            missing.append([age, missed])
    if missing:
        for age, missed in missing:
            print(f"Age group \"{age}\" missing \"{', '.join(missed)}\"")
        raise Exception("Parameters missing")


# CurrentlyInUse
# this could use some exception-catching (in fact, basically everything could)
# we're going to have a nested dictionary - age to dictionary of parameters
def readParametersAgeStructured(filename):
    agesDictionary = {}
    try:
        for line in open(filename, 'r'):
            split = line.strip().split(":")
            label = split[0].strip()
            agePar = label.split(",")
            age = agePar[0].strip()
            paramName = agePar[1].strip()
            if age not in agesDictionary:
                agesDictionary[age] = {}
            agesDictionary[age][paramName] = float(split[1].strip())
    except IndexError:
        raise Exception(f"Error: Malformed input \"{line.rstrip()}\" in {filename}") from None
    checkAgeParameters(agesDictionary)
    return agesDictionary


# CurrentlyInUse
# This needs exception-catching, and probably shouldn't have hard-coded column indices.
def readPopulationAgeStructured(filename):
    dictOfPops = {}
    boardInd = 0
    sexInd = 1
    totalAgeInd = 2
    youngInd = 3
    matureInd = 4
    oldInd = 5
    with open(filename, 'r') as f:
        first_line = f.readline()
        if first_line.strip() != "Health_Board,Sex,Total_across_age,Young,Medium,Old":
            raise ValueError("The first line must be the header")
        for n, line in enumerate(f):
            split = line.strip().split(",")
            board = split[boardInd]
            if board not in dictOfPops:
                dictOfPops[board] = {}
            sex = split[sexInd]
            if sex not in dictOfPops[board]:
                dictOfPops[board][sex] = {}
            numYoung = int(split[youngInd])
            numMature = int(split[matureInd])
            numOld = int(split[oldInd])
            numTotal = int(split[totalAgeInd])
            dictOfPops[board][sex]['y'] = numYoung
            dictOfPops[board][sex]['m'] = numMature
            dictOfPops[board][sex]['o'] = numOld
            dictOfPops[board][sex]['All_Ages'] = numTotal
            if numYoung + numMature + numOld != numTotal:
                raise ValueError(f"Line {n + 1} all ages doesn't add up")

    #     a traversal to add in the totals
    #     this is not great code, could be improved and made much more general - more robust against future age range changes
    for board in dictOfPops:
        numAllSex = 0
        numAllSexY = 0
        numAllSexM = 0
        numAllSexO = 0
        for sex in dictOfPops[board]:
            numAllSex = numAllSex + dictOfPops[board][sex]['All_Ages']
            numAllSexY = numAllSexY + dictOfPops[board][sex]['y']
            numAllSexM = numAllSexM + dictOfPops[board][sex]['m']
            numAllSexO = numAllSexO + dictOfPops[board][sex]['o']
        dictOfPops[board]['All_Sex'] = {}
        dictOfPops[board]['All_Sex']['y'] = numAllSexY
        dictOfPops[board]['All_Sex']['m'] = numAllSexM
        dictOfPops[board]['All_Sex']['o'] = numAllSexO
        dictOfPops[board]['All_Sex']['All_Ages'] = numAllSex

    return dictOfPops


# CurrentlyInUse
# making this general to include arbitrary future attributes.  Location is the primary one for right now
# keeps them in a dictionary and returns that.  Keys are
def readNodeAttributesJSON(filename):
    f = open(filename,)
    node_data = json.load(f)
    return node_data


# CurrentlyInUse
# at the moment this uses vanilla networkx edge list reading - needs weights
#  I've set it apart as its own function in case we want to do anything fancier with edge files
# in future - e.g. sampling, generating movements, whatever
# it should return a networkx graph, ideally with weighted edges
# eventual replacement with HDF5 reading code?
def genGraphFromContactFile(filename):
    G = nx.read_edgelist(filename, create_using=nx.DiGraph, delimiter=",", data=(('weight', float),))
    return G





def _check_overlap(one, two):
    """Check two AgeRange objects to see if they overlap. If they do, raise an
    Exception
    """
    if one._lower == two._lower:
        if one._upper == two._upper:
            raise Exception(f"Duplicate column header found in mixing matrix: {one}")
        raise Exception(f"Overlap in age ranges with {one} and {two}")
    if one._lower < two._lower:
        if one._upper > two._lower:
            raise Exception(f"Overlap in age ranges with {one} and {two}")
    else:
        if two._upper > one._lower:
            raise Exception(f"Overlap in age ranges with {one} and {two}")


# We use this to match an age range in the AgeRange class
AGE_RE = re.compile(r'\[(\d+),\s*(\d+)\)')


class AgeRange:
    """A helper class for an age range."""
    def __init__(self, a, b=None):
        """Initialiser. If b is None, it is assumed that a is a string to be
        parsed. Otherwise, the age range is assumed to be [a, b).
        """
        if b is None:
            if isinstance(a, tuple) and len(a) == 2:
                self._lower = a[0]
                self._upper = a[1]
            else:
                match = AGE_RE.match(a)
                if match:
                    self._lower = int(match.group(1))
                    self._upper = int(match.group(2))
                elif a[-1] == "+":
                    self._lower = int(a[:-1])
                    self._upper = -1  # A marker for no upper limit
                else:
                    raise Exception(f'Invalid age range specified: "{a}"')
            if self._upper != -1 and self._lower > self._upper:
                raise Exception(f'Invalid age range specified: {a}')
        else:
            self._lower = int(a)
            self._upper = int(b)
            if self._upper != -1 and self._lower > self._upper:
                raise Exception(f'Invalid age range specified: [{a},{b})')


    def __contains__(self, age):
        """Returns true if age is inside this age range."""
        if age < self._lower:
            return False
        if self._upper == -1:
            return True
        return age < self._upper

    def __str__(self):
        if self._upper == -1:
            return f"{self._lower}+"
        return f"[{self._lower},{self._upper})"

    def __eq__(self, other):
        return self._lower == other._lower and self._upper == other._upper

    def __neq__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self._lower, self._upper))


class MixingRow:
    """A mixing row is one row of a mixing table. This is a helper class. A row
    represents a given population, and can return the expected number of
    interactions (per day) a member of this population will have with some
    other target.
    """
    def __init__(self, ages, interactions):
        """Initialiser. Entries should be a dict mapping an AgeRange to an
        expected number of interactions."""
        self._entries = {}
        for age, interact in zip(ages, interactions):
            self._entries[age] = float(interact)

    def __getitem__(self, age):
        """Get the expected number of interactions (per day) that someone from
        this MixingRow would have with someone with the given age, or age
        range.
        """
        if isinstance(age, str) or isinstance(age, tuple):
            return self._entries[AgeRange(age)]
        for key, value in self._entries.items():
            if age in key:
                return value
        raise Exception(f'Could not find {age} in MixingRow')

    def __str__(self):
        return "[" + ", ".join(f"{str(key)}: {str(val)}"
                               for key, val in self._entries.items()) + "]"


class MixingMatrix:
    """Can give the expected number of interactions per day a given person will
    have with some other person, based on the ages of the two people, or given
    age-ranges. A MixingMatrix will often be instantiated from a CSV file (see
    initialiser). It can be used as follows:

        mm = MixingMatrix("sample_input_file/sample_20200327_comix_social_contacts.sampleCSV")
        print(mm[28][57])  # Prints the expected number of interactions a 28 year old
                           # would have with a 57 year old in a day
        print(mm["[30,40)"]["70+"])  # Prints the expected number of interactions someone in the
                                     # age range [30-40) would have with someone aged 70 or older
                                     # in any given day.
        print(mm[(30,40)]["70+"])  # As above

    """

    def __init__(self, infile):
        """Reads an input file. The input file should be a CSV file, with the
        first row and column as headers. These contain age ranges in either the
        format "[a, b)", or the format "a+". The entry in the table in row R
        and column C then indicates how many expected interactions per day a
        member of the R column interacts with a member of the C columns.
        """
        self._matrix = {}
        with open(infile, "r") as inp:
            reader = csv.reader(inp)
            headers = [AgeRange(text) for text in next(reader)[1:]]
            # Check for any overlap in the column headers
            for i, one in enumerate(headers):
                for j, two in enumerate(headers):
                    if i == j:
                        continue
                    _check_overlap(one, two)
            for row in reader:
                row_header = AgeRange(row[0])
                if row_header in self._matrix:
                    raise Exception(f"Duplicate row header found in mixing matrix: {row_header}")
                self._matrix[row_header] = MixingRow(headers, row[1:])
        # Check for any overlap in the column headers
        for one in self._matrix.keys():
            for two in self._matrix.keys():
                if one == two:
                    continue
                _check_overlap(one, two)


    def __getitem__(self, age):
        """Gets a MixingRow for the given age, which in turn can give the
        expected number of interactions. Most often, you will probably want to
        just use MixingMatrix[age1][age2] to get the expected number of
        interactions that a person of age1 will have with a person of age2.
        Note that either age1 or age2 can be numbers, or age ranges.
        """
        if isinstance(age, str) or isinstance(age, tuple):
            return self._matrix[AgeRange(age)]
        for key, value in self._matrix.items():
            if age in key:
                return value
        raise Exception(f'Could not find {age} in MixingMatrix')

    def __str__(self):
        return "\n".join(f"{str(key)}: {str(row)}" for key, row in self._matrix.items())
