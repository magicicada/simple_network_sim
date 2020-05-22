import csv
import os

import pytest


@pytest.fixture
def compartmentTransitionsByAge(compartmentTransitionsByAgeFilename):
    with open(compartmentTransitionsByAgeFilename) as fp:
        yield fp


@pytest.fixture
def compartmentTransitionsByAgeFilename():
    yield os.path.join(os.path.dirname(__file__), "..", "sample_input_files", "compartmentTransitionByAge.csv")


@pytest.fixture
def demographics(demographicsFilename):
    with open(demographicsFilename) as fp:
        yield fp


@pytest.fixture
def demographicsFilename():
    yield os.path.join(os.path.dirname(__file__), "..", "sample_input_files", "sample_hb2019_pop_est_2018_row_based.csv")


@pytest.fixture
def mixing_matrix():
    yield os.path.join(os.path.dirname(__file__), "..", "sample_input_files", "sample_20200327_comix_social_contacts.sampleCSV")


@pytest.fixture
def commute_moves():
    yield os.path.join(
        os.path.dirname(__file__), "..", "sample_input_files", "sample_scotHB_commute_moves_wu01.sampleCSV"
    )


@pytest.fixture
def locations():
    yield os.path.join(os.path.dirname(__file__), "..", "sample_input_files", "sampleNodeLocations.json")


@pytest.fixture
def compartment_names():
    yield ["S", "E", "A", "I", "H", "R", "D"]


@pytest.fixture
def age_infection_matrix(compartmentTransitionsByAge):
    ages = set()

    for row in csv.DictReader(compartmentTransitionsByAge):
        ages.add(row["age"])
    compartmentTransitionsByAge.seek(0)  # we need to seek back to the beginning since the fixture scope is function

    matrix = {}
    for a in ages:
        for b in ages:
            matrix.setdefault(a, {})[b] = 0.2

    yield matrix
