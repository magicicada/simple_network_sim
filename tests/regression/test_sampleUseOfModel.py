import pandas as pd

from simple_network_sim import sampleUseOfModel, hdf5_to_csv
from tests.utils import create_baseline


def test_cli_run(base_data_dir):
    try:
        sampleUseOfModel.main(["-c", str(base_data_dir / "config.yaml")])

        h5_file = base_data_dir / "output" / "simple_network_sim" / "outbreak-timeseries" / "data.h5"
        csv_file = base_data_dir / "output" / "simple_network_sim" / "outbreak-timeseries" / "data.csv"
        hdf5_to_csv.main([str(h5_file), str(csv_file)])
        baseline = create_baseline(csv_file)

        test_df = pd.read_csv(csv_file)
        baseline_df = pd.read_csv(baseline)

        pd.testing.assert_frame_equal(
            test_df.set_index(["date", "node", "age", "state"]),
            baseline_df.set_index(["date", "node", "age", "state"]),
            check_like=True,
        )
    finally:
        # TODO; remove this once https://github.com/ScottishCovidResponse/data_pipeline_api/issues/12 is done
        (base_data_dir / "access.log").unlink()
