import io
import json
import tempfile

import networkx as nx
import pytest

from simple_network_sim import loaders


def test_readCompartmentRatesByAge(compartmentTransitionsByAge):
    result = loaders.readCompartmentRatesByAge(compartmentTransitionsByAge)

    assert result == {
        "o": {
            "E": {"E": 0.573, "A": 0.427},
            "A": {"A": 0.803, "I": 0.0197, "R": 0.1773},
            "I": {"I": 0.67, "D": 0.0165, "H": 0.0495, "R": 0.264},
            "H": {"H": 0.9, "D": 0.042, "R": 0.058},
            "R": {"R": 1.0},
            "D": {"D": 1.0},
        },
        "m": {
            "E": {"E": 0.573, "A": 0.427},
            "A": {"A": 0.803, "I": 0.0197, "R": 0.1773},
            "I": {"I": 0.67, "D": 0.0165, "H": 0.0495, "R": 0.264},
            "H": {"H": 0.9, "D": 0.042, "R": 0.058},
            "R": {"R": 1.0},
            "D": {"D": 1.0},
        },
        "y": {
            "E": {"E": 0.573, "A": 0.427},
            "A": {"A": 0.803, "I": 0.0197, "R": 0.1773},
            "I": {"I": 0.67, "D": 0.0165, "H": 0.0495, "R": 0.264},
            "H": {"H": 0.9, "D": 0.042, "R": 0.058},
            "R": {"R": 1.0},
            "D": {"D": 1.0},
        },
    }


@pytest.mark.parametrize("contents", ["o,A,A", "o,A,0.4", "o,A,A,0.4\no,A,0.6", "A,A,1.0"])
def test_readParametersAgeStructured_missing_column(contents):
    with pytest.raises(Exception):
        contents = "age,src,dst,rate\n" + contents
        loaders.readCompartmentRatesByAge(io.StringIO(contents))


@pytest.mark.parametrize("contents", ["o,A,A,", "o,A,A,wrong"])
def test_readParametersAgeStructured_bad_value(contents):
    with pytest.raises(ValueError):
        contents = "age,src,dst,rate\n" + contents
        loaders.readCompartmentRatesByAge(io.StringIO(contents))


def test_readParametersAgeStructured_invalid_float():
    with pytest.raises(AssertionError):
        contents = "age,src,dst,rate\no,A,A,1.5\no,A,I,-0.5"
        loaders.readCompartmentRatesByAge(io.StringIO(contents))


@pytest.mark.parametrize("contents", ["o,A,A,1.0\nm,A,A,1.0", "", "age,source,destination,rate"])
def test_readParametersAgeStructured_requires_header(contents):
    with pytest.raises(AssertionError):
        assert loaders.readCompartmentRatesByAge(io.StringIO(contents))


def test_readPopulationAgeStructured(demographics):
    population = loaders.readPopulationAgeStructured(demographics)

    expected = {
        "S08000015": {
            "Female": {"y": 31950, "m": 127574, "o": 32930, "All_Ages": 192454},
            "Male": {"y": 33357, "m": 118106, "o": 25753, "All_Ages": 177216},
            "All_Sex": {"y": 65307, "m": 245680, "o": 58683, "All_Ages": 369670},
        },
        "S08000016": {
            "Female": {"y": 9957, "m": 38363, "o": 10961, "All_Ages": 59281},
            "Male": {"y": 10280, "m": 36645, "o": 9064, "All_Ages": 55989},
            "All_Sex": {"y": 20237, "m": 75008, "o": 20025, "All_Ages": 115270},
        },
        "S08000017": {
            "Female": {"y": 12176, "m": 49559, "o": 14734, "All_Ages": 76469},
            "Male": {"y": 12666, "m": 47340, "o": 12315, "All_Ages": 72321},
            "All_Sex": {"y": 24842, "m": 96899, "o": 27049, "All_Ages": 148790},
        },
        "S08000019": {
            "Female": {"y": 27102, "m": 106902, "o": 22899, "All_Ages": 156903},
            "Male": {"y": 28771, "m": 102319, "o": 18077, "All_Ages": 149167},
            "All_Sex": {"y": 55873, "m": 209221, "o": 40976, "All_Ages": 306070},
        },
        "S08000020": {
            "Female": {"y": 51441, "m": 201015, "o": 41366, "All_Ages": 293822},
            "Male": {"y": 54166, "m": 203795, "o": 32767, "All_Ages": 290728},
            "All_Sex": {"y": 105607, "m": 404810, "o": 74133, "All_Ages": 584550},
        },
        "S08000022": {
            "Female": {"y": 27187, "m": 107483, "o": 28823, "All_Ages": 163493},
            "Male": {"y": 28524, "m": 106525, "o": 23258, "All_Ages": 158307},
            "All_Sex": {"y": 55711, "m": 214008, "o": 52081, "All_Ages": 321800},
        },
        "S08000024": {
            "Female": {"y": 77589, "m": 324258, "o": 58984, "All_Ages": 460831},
            "Male": {"y": 81649, "m": 310991, "o": 44299, "All_Ages": 436939},
            "All_Sex": {"y": 159238, "m": 635249, "o": 103283, "All_Ages": 897770},
        },
        "S08000025": {
            "Female": {"y": 1847, "m": 7315, "o": 1989, "All_Ages": 11151},
            "Male": {"y": 1926, "m": 7392, "o": 1721, "All_Ages": 11039},
            "All_Sex": {"y": 3773, "m": 14707, "o": 3710, "All_Ages": 22190},
        },
        "S08000026": {
            "Female": {"y": 2112, "m": 7493, "o": 1672, "All_Ages": 11277},
            "Male": {"y": 2336, "m": 7881, "o": 1496, "All_Ages": 11713},
            "All_Sex": {"y": 4448, "m": 15374, "o": 3168, "All_Ages": 22990},
        },
        "S08000028": {
            "Female": {"y": 2204, "m": 8636, "o": 2743, "All_Ages": 13583},
            "Male": {"y": 2382, "m": 8731, "o": 2134, "All_Ages": 13247},
            "All_Sex": {"y": 4586, "m": 17367, "o": 4877, "All_Ages": 26830},
        },
        "S08000029": {
            "Female": {"y": 33155, "m": 128255, "o": 29880, "All_Ages": 191290},
            "Male": {"y": 34995, "m": 121878, "o": 23747, "All_Ages": 180620},
            "All_Sex": {"y": 68150, "m": 250133, "o": 53627, "All_Ages": 371910},
        },
        "S08000030": {
            "Female": {"y": 35016, "m": 142530, "o": 35848, "All_Ages": 213394},
            "Male": {"y": 36806, "m": 138017, "o": 27863, "All_Ages": 202686},
            "All_Sex": {"y": 71822, "m": 280547, "o": 63711, "All_Ages": 416080},
        },
        "S08000031": {
            "Female": {"y": 101619, "m": 423135, "o": 80469, "All_Ages": 605223},
            "Male": {"y": 106472, "m": 406439, "o": 56846, "All_Ages": 569757},
            "All_Sex": {"y": 208091, "m": 829574, "o": 137315, "All_Ages": 1174980},
        },
        "S08000032": {
            "Female": {"y": 61191, "m": 231416, "o": 47571, "All_Ages": 340178},
            "Male": {"y": 64096, "m": 219434, "o": 35492, "All_Ages": 319022},
            "All_Sex": {"y": 125287, "m": 450850, "o": 83063, "All_Ages": 659200},
        },
    }

    assert population == expected


def test_readPopulationAgeStructured_empty_file():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        with pytest.raises(ValueError):
            assert loaders.readPopulationAgeStructured(fp.name) == {}


def test_readPopulationAgeStructured_invalid_file():
    with pytest.raises(IOError):
        loaders.readPopulationAgeStructured("")


def test_readPopulationAgeStructured_sums_all_ages():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            "Health_Board,Sex,Total_across_age,Young,Medium,Old",
            "S08000015,Female,10,2,3,5",
            "S08000015,Male,31,7,11,13",
        ]
        fp.write("\n".join(rows))
        fp.flush()
        population = loaders.readPopulationAgeStructured(fp.name)

    assert population["S08000015"]["All_Sex"]["y"] == 2 + 7
    assert population["S08000015"]["All_Sex"]["m"] == 3 + 11
    assert population["S08000015"]["All_Sex"]["o"] == 5 + 13
    assert population["S08000015"]["All_Sex"]["All_Ages"] == 10 + 31


def test_readPopulationAgeStructured_single_row():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = ["Health_Board,Sex,Total_across_age,Young,Medium,Old", "S08000015,Female,10,2,3,5"]
        fp.write("\n".join(rows))
        fp.flush()
        population = loaders.readPopulationAgeStructured(fp.name)

    assert population["S08000015"]["Female"] == {"y": 2, "m": 3, "o": 5, "All_Ages": 10}
    assert population["S08000015"]["All_Sex"]["y"] == 2
    assert population["S08000015"]["All_Sex"]["m"] == 3
    assert population["S08000015"]["All_Sex"]["o"] == 5
    assert population["S08000015"]["All_Sex"]["All_Ages"] == 10


def test_readPopulationAgeStructured_no_rows():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = ["Health_Board,Sex,Total_across_age,Young,Medium,Old"]
        fp.write("\n".join(rows))
        fp.flush()
        assert loaders.readPopulationAgeStructured(fp.name) == {}


def test_readPopulationAgeStructured_fails_inconsistent_data():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            "Health_Board,Sex,Total_across_age,Young,Medium,Old",
            "S08000015,Female,12,2,3,5",
            "S08000015,Male,20,7,11,13",
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(ValueError):
            loaders.readPopulationAgeStructured(fp.name)


def test_readPopulationAgeStructured_fails_missing_header():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = ["S08000015,Female,12,2,3,5", "S08000015,Male,20,7,11,13"]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(ValueError):
            loaders.readPopulationAgeStructured(fp.name)


def test_readNodeAttributesJSON(locations):
    with open(locations) as fp:
        assert loaders.readNodeAttributesJSON(locations) == json.load(fp)


def test_genGraphFromContactFile(commute_moves):
    graph = nx.read_edgelist(commute_moves, create_using=nx.DiGraph, delimiter=",", data=(("weight", float),))

    assert nx.is_isomorphic(loaders.genGraphFromContactFile(commute_moves), graph)


def test_AgeRange():
    with pytest.raises(Exception) as e_info:
        a = loaders.AgeRange("[10,20)")
        assert 10 in a
        assert 15 in a
        assert 20 not in a

        b = loaders.AgeRange(10, 20)
        assert a == b
        assert 10 in b
        assert 20 not in b

        c = loaders.AgeRange((10,20))
        assert b == c

        d = loaders.AgeRange("70+")
        assert 99 in d
        assert 69 not in d

        e = loaders.AgeRange("invalid")
    assert e_info.value.args[0] == f'Invalid age range specified: "invalid"'

    with pytest.raises(Exception) as e_info:
        a = loaders.AgeRange("[10,5)")
    assert e_info.value.args[0] == f'Invalid age range specified: [10,5)'

    with pytest.raises(Exception) as e_info:
        a = loaders.AgeRange(10, 5)
    assert e_info.value.args[0] == f'Invalid age range specified: [10,5)'

    with pytest.raises(Exception) as e_info:
        a = loaders.AgeRange(20,30)
        b = loaders.AgeRange(25,35)
        loaders._check_overlap(a, b)
    assert e_info.value.args[0] == f"Overlap in age ranges with {a} and {b}"
    with pytest.raises(Exception) as e_info:
        a = loaders.AgeRange("70+")
        b = loaders.AgeRange("[65,75)")
        loaders._check_overlap(a, b)
    assert e_info.value.args[0] == f"Overlap in age ranges with {a} and {b}"


def test_invalidMixingMatrixFilesDuplicateHeaders():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,15)","[0,15)"',
            '"[0,15)",0.1, 0.05',
            '"[15,30)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            loaders.MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Duplicate column header found in mixing matrix: [0,15)"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,15)","[15,30)"',
            '"[0,15)",0.1, 0.05',
            '"[0,15)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            loaders.MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Duplicate row header found in mixing matrix: [0,15)"


def test_invalidMixingMatrixFilesOverlappingRanges():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,20)","[15,30)"',
            '"[0,15)",0.1, 0.05',
            '"[15,30)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            loaders.MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Overlap in age ranges with [0,20) and [15,30)"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,15)","[15,30)"',
            '"[0,15)",0.1, 0.05',
            '"[10,30)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            loaders.MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Overlap in age ranges with [0,15) and [10,30)"


def test_sampleMixingMatrix(mixing_matrix):
    mm = loaders.MixingMatrix(mixing_matrix)
    assert mm[28][57] == 0.097096431
    assert mm["[30,40)"][75] == 0.026352071
    assert mm[(30,40)][75] == 0.026352071
    assert mm[(30,40)][(18,30)] == 0.144896108
