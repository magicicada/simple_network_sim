# Pylint is complaining about duplicated lines, but they are all imports
# pylint: disable=duplicate-code
import shutil
from pathlib import Path

from data_pipeline_api import standard_api
from simple_network_sim import inference
import pandas as pd
import pytest


# Path to directory containing test files for fixtures
FIXTURE_DIR = Path(__file__).parents[0] / "test_data"


def _data_api(base_data_dir, config):  # pylint: disable=redefined-outer-name
    try:
        with standard_api.StandardAPI.from_config(str(base_data_dir / config), uri="", git_sha="") as store:
            yield store
    finally:
        # TODO; remove this once https://github.com/ScottishCovidResponse/SCRCIssueTracking/issues/505 is in prod
        try:
            (base_data_dir / "access.log").unlink()
        except FileNotFoundError:
            pass


@pytest.fixture
def short_simulation_dates():
    return pd.DataFrame({"Parameter": ["start_date", "end_date"], "Value": ["2020-03-16", "2020-04-16"]})


@pytest.fixture
def data_api(base_data_dir):  # pylint: disable=redefined-outer-name
    yield from _data_api(base_data_dir, "config.yaml")


@pytest.fixture
def data_api_stochastic(base_data_dir):  # pylint: disable=redefined-outer-name
    yield from _data_api(base_data_dir, "config_stochastic.yaml")


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
    shutil.rmtree(FIXTURE_DIR / "data_pipeline_inputs" / "output", ignore_errors=True)
    # Tests may drop access*.yaml files in the fixtures directory
    for logpath in (FIXTURE_DIR / "data_pipeline_inputs").glob("access*.yaml"):
        shutil.rmtree(logpath, ignore_errors=True)
    # Tests may drop access*.log files in the fixtures directory
    for logpath in (FIXTURE_DIR / "data_pipeline_inputs").glob("access*.log"):
        shutil.rmtree(logpath, ignore_errors=True)


@pytest.fixture
def abcsmc(data_api):  # pylint: disable=redefined-outer-name
    yield inference.ABCSMC(
        data_api.read_table("human/abcsmc-parameters", "abcsmc-parameters"),
        data_api.read_table("human/historical-deaths", "historical-deaths"),
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        pd.DataFrame([{"Date": "2020-01-01", "Value": 0.5}]),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/start-end-date", "start-end-date"),
        data_api.read_table("human/movement-multipliers", "movement-multipliers"),
        data_api.read_table("human/stochastic-mode", "stochastic-mode"),
        data_api.read_table("human/random-seed", "random-seed"),
    )
