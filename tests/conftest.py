import csv
import os

import pytest


@pytest.fixture
def age_transitions():
    yield os.path.join(os.path.dirname(__file__), "..", "sample_input_files", "paramsAgeStructured")


@pytest.fixture
def demographics():
    yield os.path.join(os.path.dirname(__file__), "..", "sample_input_files", "sample_hb2019_pop_est_2018.sampleCSV")

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
def age_infection_matrix(age_transitions):
    ages = set()
    with open(age_transitions) as fp:
        for row in csv.reader(fp):
            ages.add(row[0])

    matrix = {}
    for a in ages:
        for b in ages:
            matrix.setdefault(a, {})[b] = 0.2

    yield matrix
