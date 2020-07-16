import h5py
import numpy as np

from simple_network_sim.hdf5_to_csv import main


def test_hdf5_to_csv(tmp_path_factory):
    basedir = tmp_path_factory.mktemp("hdf52csv")
    csv_file = str(basedir / "hello.csv")
    h5_file = str(basedir / "hello.h5")
    h5 = h5py.File(h5_file, "w")
    array = np.array([1,2], dtype=np.dtype([("a", np.int64)]))
    h5.require_group("test").require_dataset("table", shape=array.shape, dtype=array.dtype, data=array)
    h5.close()
    main(["-c", "test", h5_file, csv_file])

    with open(csv_file) as fp:
        assert fp.read() == "a\n1\n2\n"
