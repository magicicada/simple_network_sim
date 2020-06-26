import os
import tempfile

from glob import glob
from pathlib import Path

import pandas as pd

from simple_network_sim import sampleUseOfModel
from tests.utils import create_baseline


def test_run_seeded(base_data_dir):
    try:
        sampleUseOfModel.main(["-c", str(base_data_dir / "config.yaml"), "seeded"])

        test_data = base_data_dir / "output" / "simple_network_sim" / "outbreak-timeseries" / "data.csv"
        baseline = create_baseline(test_data)

        test_df = pd.read_csv(test_data)
        baseline_df = pd.read_csv(baseline)

        pd.testing.assert_frame_equal(
            test_df.set_index(["time", "node", "age", "state"]),
            baseline_df.set_index(["time", "node", "age", "state"]),
            check_like=True,
        )
    finally:
        # TODO; remove this once https://github.com/ScottishCovidResponse/data_pipeline_api/issues/12 is done
        (base_data_dir / "access.log").unlink()
