import csv
import json
import re

import networkx as nx


# CurrentlyInUse
def _checkAgeParameters(agesDictionary):
    """
    :param agesDictionary:
    :return:

    Checks the consistency of data within the ages dictionary
    """
    all_compartments = None
    for age, compartments in agesDictionary.items():
        if all_compartments is None:
            all_compartments = list(compartments.keys())
        else:
            assert all_compartments == list(compartments.keys()), f"compartments mismatch in {age}"
        for compartment, transitions in compartments.items():
            assert compartment in transitions.keys(), f"{age},{compartment} does not have self referencing key"
            assert sum(transitions.values()) == 1.0, f"{age},{compartment} transitions do not add up to 1.0"
            for new_name, prob in transitions.items():
                assert 0.0 <= prob <= 1.0, f"{age},{compartment},{new_name},{prob} not a valid probability"

    return agesDictionary


# CurrentlyInUse
def readCompartmentRatesByAge(fp):
    """
    :param fp: file-like object that contains the age transition data
    :return: A dictionary of in the format {age: {src: {dest: prob}}}
    """
    agesDictionary = {}

    fieldnames = ["age", "src", "dst", "rate"]
    header = fp.readline().strip()
    assert header == ",".join(fieldnames), f"bad header: {header}"
    for row in csv.DictReader(fp, fieldnames=fieldnames):
        compartments = agesDictionary.setdefault(row["age"], {})
        transitions = compartments.setdefault(row["src"], {})
        transitions[row["dst"]] = float(row["rate"])

    return _checkAgeParameters(agesDictionary)


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
    assert one._upper <= two._lower or two._upper <= one._lower, \
            (f"Overlap in age ranges with {one} and {two}")


# We use this to match an age range in the AgeRange class
AGE_RE = re.compile(r'\[(\d+),\s*(\d+)\)')


class AgeRange:
    """A helper class for an age range."""


    # A marker for no upper limit
    NO_UPPER = 999999


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
                    self._upper = AgeRange.NO_UPPER
                else:
                    raise Exception(f'Invalid age range specified: "{a}"')
            if self._lower >= self._upper:
                raise Exception(f'Invalid age range specified: {a}')
        else:
            self._lower = int(a)
            self._upper = int(b)
            if self._lower >= self._upper:
                raise Exception(f'Invalid age range specified: [{a},{b})')


    def __contains__(self, age):
        """Returns true if age is inside this age range."""
        if age < self._lower:
            return False
        return age < self._upper

    def __str__(self):
        if self._upper == AgeRange.NO_UPPER:
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
                    if one == two:
                        raise Exception(f"Duplicate column header found in mixing matrix: {one}")
                    _check_overlap(one, two)
            for row in reader:
                row_header = AgeRange(row[0])
                if row_header in self._matrix:
                    raise Exception(f"Duplicate row header found in mixing matrix: {row_header}")
                self._matrix[row_header] = MixingRow(headers, row[1:])
        # Check for any overlap in the column headers
        for i, one in enumerate(self._matrix.keys()):
            for j, two in enumerate(self._matrix.keys()):
                if i == j:
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
