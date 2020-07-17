import pandas as pd


def test_load_data(data_api):
    expected = pd.DataFrame([{"Time": 0, "Value": 1.0}])
    pd.testing.assert_frame_equal(data_api.read_table("human/infection-probability"), expected)


def test_write_data(data_api, base_data_dir):
    try:
        data_api.write_table("output/data", pd.DataFrame([{"Time": 0, "Value": 1.0}]))
        df = pd.read_csv(str(base_data_dir / "output" / "data.csv"))
        pd.testing.assert_frame_equal(df, pd.DataFrame([{"Time": 0, "Value": 1.0}]))
    finally:
        try:
            # TODO: make this into a fixture if we need to test the DataAPI any more heavily
            (base_data_dir / "output" / "data").unlink()
        except FileNotFoundError:
            pass
