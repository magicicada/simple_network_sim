"""A mixing matrix descibes how various groups of a population mix together.
"""

import csv
import re

AGE_RE = re.compile(r'\[(\d+),\s*(\d+)\)')


def _check_overlap(one, two):
    """Check two AgeRange objects to see if they overlap. If they do, raise an
    Exception
    """
    if one._lower == two._lower:
        raise Exception(f"Overlap in age ranges with {one} and {two}")
    if one._lower < two._lower:
        if one._upper > two._lower:
            raise Exception(f"Overlap in age ranges with {one} and {two}")
    else:
        if two._upper > one._lower:
            raise Exception(f"Overlap in age ranges with {one} and {two}")


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
            if len(headers) != len(set(headers)):
                raise Exception("Duplicate header found in mixing matrix")
            # Check for any overlap in the age ranges
            for one, two in zip(headers, headers):
                if one == two:
                    continue
                _check_overlap(one, two)
            for row in reader:
                self._matrix[AgeRange(row[0])] = MixingRow(headers, row[1:])


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
