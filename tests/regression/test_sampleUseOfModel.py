import pandas as pd

from simple_network_sim import network_of_populations, sampleUseOfModel, hdf5_to_csv
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
        h5_file.unlink()
        csv_file.unlink()


def test_stochastic_cli_run(base_data_dir):
    try:
        sampleUseOfModel.main(["-c", str(base_data_dir / "config_stochastic.yaml")])

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
        h5_file.unlink()
        csv_file.unlink()


def test_stochastic_seed_sequence(data_api_stochastic):
    network, _ = network_of_populations.createNetworkOfPopulation(
        data_api_stochastic.read_table("human/compartment-transition", "compartment-transition"),
        data_api_stochastic.read_table("human/population", "population"),
        data_api_stochastic.read_table("human/commutes", "commutes"),
        data_api_stochastic.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api_stochastic.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api_stochastic.read_table("human/infection-probability", "infection-probability"),
        data_api_stochastic.read_table("human/initial-infections", "initial-infections"),
        pd.DataFrame({"Value": [2]}),
        data_api_stochastic.read_table("human/start-end-date", "start-end-date"),
        data_api_stochastic.read_table("human/movement-multipliers", "movement-multipliers"),
        pd.DataFrame({"Value": [True]}),
    )

    issues = []
    r1, r2 = sampleUseOfModel.runSimulation(network, random_seed=123, issues=issues, max_workers=2)
    df1 = r1.output
    df2 = r2.output

    # It's very unlikely these numbers would match unless both runs produce the same numbers
    assert df1[df1.state == "D"].total.sum() != df2[df2.state == "D"].total.sum()
    assert not issues
