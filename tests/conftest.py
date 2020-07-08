import shutil
import tempfile
from pathlib import Path

import pytest
from simple_network_sim.data import Datastore

# Path to directory containing test files for fixtures
FIXTURE_DIR = Path(__file__).parents[0] / "test_data"


@pytest.fixture
def data_api(base_data_dir):
    try:
        with Datastore(str(base_data_dir / "config.yaml")) as store:
            yield store
    finally:
        # TODO; remove this once https://github.com/ScottishCovidResponse/data_pipeline_api/issues/12 is done
        try:
            (base_data_dir / "access.log").unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def base_data_dir():
    yield FIXTURE_DIR / "data_pipeline_inputs"


@pytest.fixture
def locations():
    yield FIXTURE_DIR / "sampleNodeLocations.json"


@pytest.fixture(autouse=True, scope="session")
def teardown_remove_data():
    """Remove test output created during testing.

    Datasets defined in data_pipeline_inputs/config.yaml can't be handled with
    pytest's tmp_path, so are cleaned up here. Change these locations as necessary
    when the config file changes.
    """
    yield
    shutil.rmtree(FIXTURE_DIR / "data_pipeline_inputs" / "output")
    # Tests may drop access*.yaml files in the fixtures directory
    for logpath in (FIXTURE_DIR / "data_pipeline_inputs").glob("access*.yaml"):
        shutil.rmtree(logpath)
    # Tests may drop access*.log files in the fixtures directory
    for logpath in (FIXTURE_DIR / "data_pipeline_inputs").glob("access*.log"):
        shutil.rmtree(logpath)
