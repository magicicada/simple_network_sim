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
        "S08000015": {"y": 65307, "m": 245680, "o": 58683},
        "S08000016": {"y": 20237, "m": 75008, "o": 20025},
        "S08000017": {"y": 24842, "m": 96899, "o": 27049},
        "S08000019": {"y": 55873, "m": 209221, "o": 40976},
        "S08000020": {"y": 105607, "m": 404810, "o": 74133},
        "S08000022": {"y": 55711, "m": 214008, "o": 52081},
        "S08000024": {"y": 159238, "m": 635249, "o": 103283},
        "S08000025": {"y": 3773, "m": 14707, "o": 3710},
        "S08000026": {"y": 4448, "m": 15374, "o": 3168},
        "S08000028": {"y": 4586, "m": 17367, "o": 4877},
        "S08000029": {"y": 68150, "m": 250133, "o": 53627},
        "S08000030": {"y": 71822, "m": 280547, "o": 63711},
        "S08000031": {"y": 208091, "m": 829574, "o": 137315},
        "S08000032": {"y": 125287, "m": 450850, "o": 83063},
    }

    assert population == expected


def test_readPopulationAgeStructured_empty_file():
    with pytest.raises(AssertionError):
        assert loaders.readPopulationAgeStructured(io.StringIO("")) == {}


@pytest.mark.parametrize("header", ["Health_Board,Sex,Total", "Total"])
def test_readPopulationAgeStructured_bad_header(header):
    with pytest.raises(AssertionError):
        loaders.readPopulationAgeStructured(io.StringIO(f"{header}\nS08000015,Female,o,10"))


@pytest.mark.parametrize("data", ["S08000015,Female,o", "S08000015,Female,o,"])
def test_readPopulationAgeStructured_all_columns_required(data):
    with pytest.raises(AssertionError):
        loaders.readPopulationAgeStructured(io.StringIO(f"Health_Board,Sex,Age,Total\n{data}"))


@pytest.mark.parametrize("total", ["-20", "NaN", "ten"])
def test_readPopulationAgeStructured_bad_total(total):
    rows = [
        "Health_Board,Sex,Age,Total",
        f"S08000015,Female,y,{total}",
    ]

    with pytest.raises(ValueError):
        loaders.readPopulationAgeStructured(io.StringIO("\n".join(rows)))


def test_readPopulationAgeStructured_aggregate_ages():
    rows = ["Health_Board,Sex,Age,Total", "S08000015,Female,o,100", "S08000015,Male,o,100"]
    population = loaders.readPopulationAgeStructured(io.StringIO("\n".join(rows)))

    assert population == {"S08000015": {"o": 200}}


def test_readNodeAttributesJSON(locations):
    with open(locations) as fp:
        assert loaders.readNodeAttributesJSON(locations) == json.load(fp)


def test_genGraphFromContactFile(commute_moves):
    graph = nx.read_edgelist(commute_moves, create_using=nx.DiGraph, delimiter=",", data=(("weight", float),))

    assert nx.is_isomorphic(loaders.genGraphFromContactFile(commute_moves), graph)


def test_AgeRange():
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


@pytest.mark.parametrize("invalid_range", ["[10,300)", "200+", "[10,10)", "[10,5)"])
def test_AgeRange_invalid_range_str(invalid_range):
    with pytest.raises(AssertionError):
        loaders.AgeRange(invalid_range)


@pytest.mark.parametrize("invalid_range", [(10, 300), (10, 10), (10, 5), (-10, 20)])
def test_AgeRange_invalid_range_tuple(invalid_range):
    with pytest.raises(AssertionError):
        loaders.AgeRange(invalid_range)
    with pytest.raises(AssertionError):
        loaders.AgeRange(*invalid_range)


@pytest.mark.parametrize("wrong_value", ["invalid", "(0,10)", "[0,10]", "[a,b)", "[-10,20)"])
def test_AgeRange_wrong_format_str(wrong_value):
    with pytest.raises(Exception) as e_info:
        loaders.AgeRange(wrong_value)
    assert e_info.value.args[0] == f'Invalid age range specified: "{wrong_value}"'


@pytest.mark.parametrize("wrong_value", [10, (20,)])
def test_AgeRange_wrong_format_other(wrong_value):
    with pytest.raises(Exception) as e_info:
        loaders.AgeRange(wrong_value)


@pytest.mark.parametrize("range_a,range_b", [((0, 10), "[0,10)"), ("[0,10)", "[0,10)"), ("70+", "70+"), ("70+", "[70,200)"), ("70+", (70, 200))])
def test_AgeRange_equivalent_ranges(range_a, range_b):
    a = loaders.AgeRange(range_a)
    b = loaders.AgeRange(range_b)

    assert hash(a) == hash(b)
    assert a == b
    assert not (a != b)


def test_AgeRange_equivalent_ranges_params():
    assert hash(loaders.AgeRange(0, 10)) == hash(loaders.AgeRange(0, 10)) and loaders.AgeRange(0, 10) == loaders.AgeRange(0, 10)
    assert hash(loaders.AgeRange(0, 10)) == hash(loaders.AgeRange((0, 10))) and loaders.AgeRange(0, 10) == loaders.AgeRange((0, 10))
    assert hash(loaders.AgeRange(0, 10)) == hash(loaders.AgeRange("[0,10)")) and loaders.AgeRange(0, 10) == loaders.AgeRange("[0, 10)")


@pytest.mark.parametrize("range_a,range_b", [((0, 10), "[0,11)"), ("[1,10)", "[0,10)"), ("71+", "70+"), ("70+", "[70,199)"), ("70+", (70, 199))])
def test_AgeRange_different_ranges(range_a, range_b):
    a = loaders.AgeRange(range_a)
    b = loaders.AgeRange(range_b)

    assert hash(a) != hash(b)
    assert a != b
    assert not (a == b)


@pytest.mark.parametrize("range,expected", [((0, 10), "[0,10)"), ("[0,10)", "[0,10)"), ("70+", "70+"), ((70, 200), "70+")])
def test_AgeRange_to_str(range, expected):
    assert str(loaders.AgeRange(range)) == expected


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
