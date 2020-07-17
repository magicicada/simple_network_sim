import json

from data_pipeline_api.file_formats import object_file
import networkx as nx
import pandas as pd
import pytest

from simple_network_sim import loaders


def test_readCompartmentRatesByAge(data_api):
    result = loaders.readCompartmentRatesByAge(data_api.read_table("human/compartment-transition", "compartment-transition"))

    assert result == {
        "70+": {
            "E": {"E": pytest.approx(0.573), "A": pytest.approx(0.427)},
            "A": {"A": pytest.approx(0.803), "I": pytest.approx(0.0197), "R": pytest.approx(0.1773)},
            "I": {"I": pytest.approx(0.67), "D": pytest.approx(0.0165), "H": pytest.approx(0.0495), "R": pytest.approx(0.264)},
            "H": {"H": pytest.approx(0.9), "D": pytest.approx(0.042), "R": pytest.approx(0.058)},
            "R": {"R": pytest.approx(1.0)},
            "D": {"D": pytest.approx(1.0)},
        },
        "[17,70)": {
            "E": {"E": pytest.approx(0.573), "A": pytest.approx(0.427)},
            "A": {"A": pytest.approx(0.803), "I": pytest.approx(0.0197), "R": pytest.approx(0.1773)},
            "I": {"I": pytest.approx(0.67), "D": pytest.approx(0.0165), "H": pytest.approx(0.0495), "R": pytest.approx(0.264)},
            "H": {"H": pytest.approx(0.9), "D": pytest.approx(0.042), "R": pytest.approx(0.058)},
            "R": {"R": pytest.approx(1.0)},
            "D": {"D": pytest.approx(1.0)},
        },
        "[0,17)": {
            "E": {"E": pytest.approx(0.573), "A": pytest.approx(0.427)},
            "A": {"A": pytest.approx(0.803), "I": pytest.approx(0.0197), "R": pytest.approx(0.1773)},
            "I": {"I": pytest.approx(0.67), "D": pytest.approx(0.0165), "H": pytest.approx(0.0495), "R": pytest.approx(0.264)},
            "H": {"H": pytest.approx(0.9), "D": pytest.approx(0.042), "R": pytest.approx(0.058)},
            "R": {"R": pytest.approx(1.0)},
            "D": {"D": pytest.approx(1.0)},
        },
    }


def test_readCompartmentRatesByAge_approximately_one():
    result = loaders.readCompartmentRatesByAge(pd.DataFrame([{"age": "70+", "src": "A", "dst": "A", "rate": 0.999999999}]))

    assert result == {"70+": {"A": {"A": 0.999999999}}}


def test_readParametersAgeStructured_invalid_float():
    with pytest.raises(AssertionError):
        df = pd.DataFrame([
            {"age": "70+", "src": "A", "dst": "A", "rate": 1.5},
            {"age": "70+", "src": "A", "dst": "I", "rate": -0.5},
        ])
        loaders.readCompartmentRatesByAge(df)


def test_readParametersAgeStructured_more_than_100_percent():
    with pytest.raises(AssertionError):
        df = pd.DataFrame([
            {"age": "70+", "src": "A", "dst": "A", "rate": 0.7},
            {"age": "70+", "src": "A", "dst": "I", "rate": 0.5},
        ])
        loaders.readCompartmentRatesByAge(df)


def test_readParametersAgeStructured_less_than_100_percent():
    with pytest.raises(AssertionError):
        df = pd.DataFrame([
            {"age": "70+", "src": "A", "dst": "A", "rate": 0.5},
            {"age": "70+", "src": "A", "dst": "I", "rate": 0.2},
        ])
        loaders.readCompartmentRatesByAge(df)


def test_readPopulationAgeStructured(data_api):
    population = loaders.readPopulationAgeStructured(data_api.read_table("human/population", "population"))

    expected = {
        "S08000015": {"[0,17)": 65307, "[17,70)": 245680, "70+": 58683},
        "S08000016": {"[0,17)": 20237, "[17,70)": 75008, "70+": 20025},
        "S08000017": {"[0,17)": 24842, "[17,70)": 96899, "70+": 27049},
        "S08000019": {"[0,17)": 55873, "[17,70)": 209221, "70+": 40976},
        "S08000020": {"[0,17)": 105607, "[17,70)": 404810, "70+": 74133},
        "S08000022": {"[0,17)": 55711, "[17,70)": 214008, "70+": 52081},
        "S08000024": {"[0,17)": 159238, "[17,70)": 635249, "70+": 103283},
        "S08000025": {"[0,17)": 3773, "[17,70)": 14707, "70+": 3710},
        "S08000026": {"[0,17)": 4448, "[17,70)": 15374, "70+": 3168},
        "S08000028": {"[0,17)": 4586, "[17,70)": 17367, "70+": 4877},
        "S08000029": {"[0,17)": 68150, "[17,70)": 250133, "70+": 53627},
        "S08000030": {"[0,17)": 71822, "[17,70)": 280547, "70+": 63711},
        "S08000031": {"[0,17)": 208091, "[17,70)": 829574, "70+": 137315},
        "S08000032": {"[0,17)": 125287, "[17,70)": 450850, "70+": 83063},
    }

    assert population == expected


@pytest.mark.parametrize("total", ["-20", "NaN", "ten"])
def test_readPopulationAgeStructured_bad_total(total):
    df = pd.DataFrame([
        {"Health_Board": "S08000015", "Sex": "Female", "Age": "[17,70)", "Total": total},
    ])

    with pytest.raises(ValueError):
        loaders.readPopulationAgeStructured(df)


def test_readPopulationAgeStructured_aggregate_ages():
    df = pd.DataFrame([
        {"Health_Board": "S08000015", "Sex": "Female", "Age": "[17,70)", "Total": 100},
        {"Health_Board": "S08000015", "Sex": "Male", "Age": "[17,70)", "Total": 100},
    ])
    population = loaders.readPopulationAgeStructured(df)

    assert population == {"S08000015": {"[17,70)": 200}}


def test_readInfectiousStates():
    assert set(loaders.readInfectiousStates(pd.DataFrame([{"Compartment": "A"}, {"Compartment": "I"}]))) == {"A", "I"}


def test_readInfectiousStates_empty():
    assert loaders.readInfectiousStates(pd.DataFrame([])) == []


@pytest.mark.parametrize("invalid_infected", ["asdf", float("NaN"), -1, float("inf")])
def test_readInitialInfections_invalid_total(invalid_infected):
    df = pd.DataFrame([("S08000015", "[17,70)", invalid_infected)], columns=("Health_Board", "Age", "Infected"))
    with pytest.raises(ValueError):
        loaders.readInitialInfections(df)


def test_readInitialInfections():
    df = pd.DataFrame(
        [("S08000015", "[17,70)", 10), ("S08000015", "70+", 5), ("S08000016", "70+", 5)],
        columns=("Health_Board", "Age", "Infected"),
    )
    infected = loaders.readInitialInfections(df)

    assert infected == {"S08000015": {"[17,70)": 10.0, "70+": 5.0}, "S08000016": {"70+": 5.0}}


def test_readNodeAttributesJSON(locations):
    with open(locations) as fp:
        assert loaders.readNodeAttributesJSON(locations) == json.load(fp)


def test_genGraphFromContactFile(base_data_dir, data_api):
    with open(str(base_data_dir / "human" / "commutes" / "1" / "data.h5"), "rb") as fp:
        df = object_file.read_table(fp, "commutes")
    graph = nx.convert_matrix.from_pandas_edgelist(df, edge_attr=True, create_using=nx.DiGraph)

    assert nx.is_isomorphic(loaders.genGraphFromContactFile(data_api.read_table("human/commutes", "commutes")), graph)


def test_genGraphFromContactFile_negative_delta_adjustment():
    df = pd.DataFrame([
        {"source": "a", "target": "b", "weight": 0, "delta_adjustment": -1.0}
    ])
    with pytest.raises(AssertionError):
        loaders.genGraphFromContactFile(df)


def test_genGraphFromContactFile_negative_weight():
    df = pd.DataFrame([
        {"source": "a", "target": "b", "weight": -30.0, "delta_adjustment": 1.0}
    ])

    with pytest.raises(AssertionError):
        loaders.genGraphFromContactFile(df)


def test_readMovementMultipliers(data_api):
    ms = loaders.readMovementMultipliers(data_api.read_table("human/movement-multipliers", "movement-multipliers"))

    assert ms == {
        50: loaders.Multiplier(movement=0.05, contact=0.05),
        75: loaders.Multiplier(movement=0.3, contact=0.3),
        80: loaders.Multiplier(movement=0.8, contact=0.8),
        100: loaders.Multiplier(movement=0.9, contact=0.9)
    }


@pytest.mark.parametrize("m", [float("NaN"), float("inf"), -1.0, "asdf"])
def test_readMovementMultipliers_bad_movement_multipliers(m):
    df = pd.DataFrame([{"Time": 0, "Movement_Multiplier": m, "Contact_Multiplier": 1.0}])
    with pytest.raises(ValueError):
        loaders.readMovementMultipliers(df)


@pytest.mark.parametrize("m", [float("NaN"), float("inf"), -1.0, "asdf"])
def test_readMovementMultipliers_bad_contact_multipliers(m):
    df = pd.DataFrame([{"Time": 0, "Movement_Multiplier": 1.0, "Contact_Multiplier": m}])
    with pytest.raises(ValueError):
        loaders.readMovementMultipliers(df)


@pytest.mark.parametrize("t", [-1, "asdf"])
def test_readMovementMultipliers_bad_times(t):
    df = pd.DataFrame([{"Time": t, "Movement_Multiplier": 1.0, "Contact_Multiplier": 1.0}])
    with pytest.raises(ValueError):
        loaders.readMovementMultipliers(df)


def test_readInfectionProbability():
    df = pd.DataFrame([(0, 0.3), (4, 0.7), (30, 1.0)], columns=["Time", "Value"])

    assert loaders.readInfectionProbability(df) == {0: 0.3, 4: 0.7, 30: 1.0}


def test_readInfectionProbability_empty():
    with pytest.raises(ValueError):
        loaders.readInfectionProbability(pd.DataFrame())


def test_readInfectionProbability_no_zeroth_time():
    with pytest.raises(ValueError):
        loaders.readInfectionProbability(pd.DataFrame([{"Time": 3, "Value": 1.0}]))


def test_readInfectionProbability_invalid_time():
    with pytest.raises(ValueError):
        loaders.readInfectionProbability(pd.DataFrame([{"Time": -1, "Value": 1.0}, {"Time": 0, "Value": 1.0}]))


@pytest.mark.parametrize("prob", [-1.0, 2.0, float("inf"), float("nan"), "asdf"])
def test_readInfectionProbability_invalid_prob(prob):
    with pytest.raises(ValueError):
        loaders.readInfectionProbability(pd.DataFrame([{"Time": 0, "Value": prob}]))


def test_readRandomSeed():
    df = pd.DataFrame([123], columns=["Value"])

    assert loaders.readRandomSeed(df) == 123


def test_readRandomSeed_bad_shape():
    with pytest.raises(AssertionError):
        loaders.readRandomSeed(pd.DataFrame())

    with pytest.raises(AssertionError):
        loaders.readRandomSeed(pd.DataFrame([0, 4, 30], columns=["Value"]))

    with pytest.raises(AssertionError):
        loaders.readRandomSeed(pd.DataFrame([0], columns=["Bad name"]))


def test_readInfectionProbability_invalid_seed():
    with pytest.raises(ValueError):
        loaders.readRandomSeed(pd.DataFrame([1.2], columns=["Value"]))

    with pytest.raises(ValueError):
        loaders.readRandomSeed(pd.DataFrame([-1], columns=["Value"]))


def test_readTrials():
    df = pd.DataFrame([1000], columns=["Value"])

    assert loaders.readTrials(df) == 1000


def test_readTrials_bad_shape():
    with pytest.raises(AssertionError):
        loaders.readTrials(pd.DataFrame())

    with pytest.raises(AssertionError):
        loaders.readTrials(pd.DataFrame([0, 4, 30], columns=["Value"]))

    with pytest.raises(AssertionError):
        loaders.readTrials(pd.DataFrame([0], columns=["Bad name"]))


def test_readTrials_invalid_trials():
    with pytest.raises(ValueError):
        loaders.readTrials(pd.DataFrame([1.2], columns=["Value"]))

    with pytest.raises(ValueError):
        loaders.readTrials(pd.DataFrame([-1], columns=["Value"]))


def test_readStochasticMode():
    df = pd.DataFrame([True], columns=["Value"])

    assert loaders.readStochasticMode(df) is True


def test_readStochasticMode_bad_shape():
    with pytest.raises(AssertionError):
        loaders.readStochasticMode(pd.DataFrame())

    with pytest.raises(AssertionError):
        loaders.readStochasticMode(pd.DataFrame([True, False, True], columns=["Value"]))

    with pytest.raises(AssertionError):
        loaders.readStochasticMode(pd.DataFrame([0], columns=["Bad name"]))


def test_readStochasticMode_invalid_trials():
    with pytest.raises(ValueError):
        loaders.readStochasticMode(pd.DataFrame(["True"], columns=["Value"]))

    with pytest.raises(ValueError):
        loaders.readStochasticMode(pd.DataFrame([10], columns=["Value"]))


def test_AgeRange():
    a = loaders.AgeRange("[10,20)")
    assert a.age_group == "[10,20)"


def test_AgeRange_hash():
    a = loaders.AgeRange("[10,20)")
    assert hash(a) == hash(a.age_group)


def test_AgeRange_equal():
    assert loaders.AgeRange("[10,20)") == loaders.AgeRange("[10,20)")
    assert not loaders.AgeRange("[10,20)") == loaders.AgeRange("[10,21)")


def test_AgeRange_str():
    age = "[10,20)"
    a = loaders.AgeRange(age)
    assert str(a) is age


@pytest.mark.parametrize("range_a,range_b", [((0, 10), "[0,11)"), ("[1,10)", "[0,10)"), ("71+", "70+"), ("70+", "[70,199)"), ("70+", (70, 199))])
def test_AgeRange_different_ranges(range_a, range_b):
    a = loaders.AgeRange(range_a)
    b = loaders.AgeRange(range_b)

    assert hash(a) != hash(b)
    assert a != b
    assert not (a == b)


def test_sampleMixingMatrix(data_api):
    mm = loaders.MixingMatrix(data_api.read_table("human/mixing-matrix", "mixing-matrix"))
    assert mm["[17,70)"]["[17,70)"] == 0.2


def test_sampleMixingMatrix_iterate_keys(data_api):
    matrix = loaders.MixingMatrix(data_api.read_table("human/mixing-matrix", "mixing-matrix"))
    assert set(matrix) == {"[0,17)", "[17,70)", "70+"}
    for key in matrix:
        assert matrix[key]


def test_sampleMixingMatrix_iterate_keys_one_element():
    df = pd.DataFrame([{"source": "[0,15)", "target": "[0,15)", "mixing": 0.1}])
    matrix = loaders.MixingMatrix(df)
    assert list(matrix) == ["[0,15)"]
    for key in matrix:
        assert matrix[key]


def test_MixingRow_iterate_over_keys():
    row = loaders.MixingRow(["[0,17)", "[17,70)", "70+"], ["0.2", "0.03", "0.1"])
    assert list(row) == ["[0,17)", "[17,70)", "70+"]
    for key in row:
        assert row[key] in [0.2, 0.03, 0.1]


def test_MixingRow_iterate_access_key():
    row = loaders.MixingRow(["[0,17)", "[17,70)", "70+"], ["0.2", "0.03", "0.1"])

    assert row["[0,17)"] == 0.2
    assert row["[17,70)"] == 0.03
    assert row["70+"] == 0.1


def test_MixingRow_str():
    row = loaders.MixingRow(["[0,17)", "[17,70)", "70+"], ["0.2", "0.03", "0.1"])

    assert str(row) == "[[0,17): 0.2, [17,70): 0.03, 70+: 0.1]"
