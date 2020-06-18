import csv
import tempfile

from pathlib import Path

import pytest

# Path to directory containing test files for fixtures
from data_pipeline_api.api import API
from data_pipeline_api.file_system_data_access import FileSystemDataAccess

FIXTURE_DIR = Path(__file__).parents[0] / "test_data"


@pytest.fixture
def data_api(base_data_dir):
    with tempfile.NamedTemporaryFile(delete=False) as metadata:
        yield API(FileSystemDataAccess(base_data_dir, metadata.name))


@pytest.fixture
def base_data_dir():
    yield FIXTURE_DIR / "data_pipeline_inputs"


@pytest.fixture
def locations():
    yield FIXTURE_DIR / "sampleNodeLocations.json"
