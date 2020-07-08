import tempfile
from pathlib import Path

import pytest
from simple_network_sim.data import Datastore

# Path to directory containing test files for fixtures
FIXTURE_DIR = Path(__file__).parents[0] / "test_data"


def _data_api(base_data_dir, config):
    try:
        with Datastore(str(base_data_dir / config)) as store:
            yield store
    finally:
        # TODO; remove this once https://github.com/ScottishCovidResponse/SCRCIssueTracking/issues/505 is in prod
        try:
            (base_data_dir / "access.log").unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def data_api(base_data_dir):
    yield from _data_api(base_data_dir, "config.yaml")


@pytest.fixture
def data_api_stochastic(base_data_dir):
    yield from _data_api(base_data_dir, "config_stochastic.yaml")


@pytest.fixture
def base_data_dir():
    yield FIXTURE_DIR / "data_pipeline_inputs"


@pytest.fixture
def locations():
    yield FIXTURE_DIR / "sampleNodeLocations.json"
