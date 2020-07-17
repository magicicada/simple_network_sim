import h5py

from simple_network_sim.csv_to_hdf5 import main


def test_csv_to_hdf5(tmp_path_factory):
    basedir = tmp_path_factory.mktemp("csv2hdf5")
    csv_file = str(basedir / "hello.csv")
    h5_file = str(basedir / "hello.h5")
    with open(csv_file, "w") as fp:
        fp.write("A,B\n1.0,0.5\n")
    main(["-c", "test", csv_file, h5_file])

    h5 = h5py.File(h5_file, "r")
    assert h5["test"]["table"][()].tolist() == [(1.0, 0.5)]
