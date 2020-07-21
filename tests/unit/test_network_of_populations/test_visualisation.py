from pathlib import Path

import pandas as pd
import pytest
from data_pipeline_api.file_formats import object_file

import simple_network_sim.network_of_populations.visualisation as vis
from tests.utils import compare_mpl_plots

def test_plotStates_three_rows():
    simple = pd.DataFrame([
        {"date": "2020-04-12", "node": "hb1", "state": "S", "total": 15.0},
        {"date": "2020-04-12", "node": "hb2", "state": "S", "total": 21.0},
        {"date": "2020-04-12", "node": "hb3", "state": "S", "total": 20.0},
        {"date": "2020-04-12", "node": "hb3", "state": "E", "total": 0.0},
        {"date": "2020-04-12", "node": "hb4", "state": "S", "total": 10.0},
        {"date": "2020-04-12", "node": "hb5", "state": "S", "total": 10.0},
        {"date": "2020-04-12", "node": "hb6", "state": "S", "total": 10.0},
        {"date": "2020-04-12", "node": "hb7", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb1", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb2", "state": "S", "total": 5.0},
        {"date": "2020-04-13", "node": "hb3", "state": "S", "total": 5.0},
        {"date": "2020-04-13", "node": "hb3", "state": "E", "total": 15.0},
        {"date": "2020-04-13", "node": "hb4", "state": "S", "total": 0.0},
        {"date": "2020-04-13", "node": "hb5", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb6", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb7", "state": "S", "total": 10.0},
    ])
    compare_mpl_plots(vis.plot_nodes(pd.DataFrame(simple)))


def test_plotStates_two_rows():
    simple = pd.DataFrame([
        {"date": "2020-04-12", "node": "hb1", "state": "S", "total": 15.0},
        {"date": "2020-04-12", "node": "hb2", "state": "S", "total": 21.0},
        {"date": "2020-04-12", "node": "hb3", "state": "S", "total": 20.0},
        {"date": "2020-04-12", "node": "hb3", "state": "E", "total": 0.0},
        {"date": "2020-04-12", "node": "hb4", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb1", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb2", "state": "S", "total": 5.0},
        {"date": "2020-04-13", "node": "hb3", "state": "S", "total": 5.0},
        {"date": "2020-04-13", "node": "hb3", "state": "E", "total": 15.0},
        {"date": "2020-04-13", "node": "hb4", "state": "S", "total": 0.0},
    ])
    compare_mpl_plots(vis.plot_nodes(pd.DataFrame(simple)))


def test_plotStates_single_row():
    simple = pd.DataFrame([
        {"date": "2020-04-12", "node": "hb1", "state": "S", "total": 15.0},
        {"date": "2020-04-12", "node": "hb2", "state": "S", "total": 21.0},
        {"date": "2020-04-13", "node": "hb1", "state": "S", "total": 10.0},
        {"date": "2020-04-13", "node": "hb2", "state": "S", "total": 5.0},
    ])
    compare_mpl_plots(vis.plot_nodes(pd.DataFrame(simple)))


def test_plotStates_empty_node():
    simple = pd.DataFrame([{"time": 0, "node": "hb1", "state": "S", "total": 15.0}])
    with pytest.raises(ValueError):
        vis.plot_nodes(pd.DataFrame(simple), nodes=[])


def test_plotStates_empty_states():
    simple = pd.DataFrame([{"time": 0, "node": "hb1", "state": "S", "total": 15.0}])
    with pytest.raises(ValueError):
        vis.plot_nodes(pd.DataFrame(simple), states=[])


def test_plotStates_empty_missing_column():
    simple = pd.DataFrame([{"node": "hb1", "state": "S", "total": 15.0}])
    with pytest.raises(ValueError):
        vis.plot_nodes(pd.DataFrame(simple), states=[])


def test_read_output(tmpdir):
    df = pd.DataFrame([{"a": 10, "b": 20}])
    with open(str(tmpdir / Path("simple.h5")), "wb") as fp:
        object_file.write_table(fp, "outbreak-timeseries", df)
    path = str(tmpdir / Path("access.yaml"))
    with open(path, "w") as fp:
        fp.write(f"""
data_directory: {str(tmpdir)}
io:
- type: write
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
    filename: simple.h5
""")
    output = vis.read_output("output/simple_network_sim/outbreak-timeseries", path)

    pd.testing.assert_frame_equal(output, df)


def test_read_output_ignore_read(tmpdir):
    df = pd.DataFrame([{"a": 10, "b": 20}])
    with open(str(tmpdir / Path("simple.h5")), "wb") as fp:
        object_file.write_table(fp, "outbreak-timeseries", df)
    path = str(tmpdir / Path("access.yaml"))
    with open(path, "w") as fp:
        fp.write(f"""
data_directory: {str(tmpdir)}
io:
- type: read
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
- type: write
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
    filename: simple.h5
""")
    output = vis.read_output("output/simple_network_sim/outbreak-timeseries", path)

    pd.testing.assert_frame_equal(output, df)


def test_read_output_multiple_writes(tmpdir):
    df = pd.DataFrame([{"a": 10, "b": 20}])
    with open(str(tmpdir / Path("simple.h5")), "wb") as fp:
        object_file.write_table(fp, "outbreak-timeseries", df)
    path = str(tmpdir / Path("access.yaml"))
    with open(path, "w") as fp:
        fp.write(f"""
data_directory: {str(tmpdir)}
io:
- type: write
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
    filename: simple.h5
- type: write
  call_metadata:
    data_product: other
  access_metadata:
    data_product: other
    filename: simple.h5
""")
    output = vis.read_output("output/simple_network_sim/outbreak-timeseries", path)

    pd.testing.assert_frame_equal(output, df)


def test_read_output_duplicated(tmpdir):
    df = pd.DataFrame([{"a": 10, "b": 20}])
    df.to_csv(str(tmpdir / Path("simple.csv")), index=False)
    path = str(tmpdir / Path("access.yaml"))
    with open(path, "w") as fp:
        fp.write(f"""
data_directory: {str(tmpdir)}
io:
- type: write
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
    filename: simple.csv
- type: write
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
    filename: simple.csv
""")
    with pytest.raises(AssertionError):
        vis.read_output("output/simple_network_sim/outbreak-timeseries", path)


def test_read_output_no_writes(tmpdir):
    df = pd.DataFrame([{"a": 10, "b": 20}])
    df.to_csv(str(tmpdir / Path("simple.csv")), index=False)
    path = str(tmpdir / Path("access.yaml"))
    with open(path, "w") as fp:
        fp.write(f"""
data_directory: {str(tmpdir)}
io:
- type: read
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
    filename: simple.csv
""")
    with pytest.raises(AssertionError):
        vis.read_output("output/simple_network_sim/outbreak-timeseries", path)


def test_read_output_use_call_metadata_data_product(tmpdir):
    df = pd.DataFrame([{"a": 10, "b": 20}])
    df.to_csv(str(tmpdir / Path("simple.csv")), index=False)
    path = str(tmpdir / Path("access.yaml"))
    with open(path, "w") as fp:
        fp.write(f"""
data_directory: {str(tmpdir)}
io:
- type: read
  call_metadata:
    data_product: output/simple_network_sim/outbreak-timeseries
  access_metadata:
    data_product: wrong
    filename: simple.csv
""")
    with pytest.raises(AssertionError):
        vis.read_output("output/simple_network_sim/outbreak-timeseries", path)
