import tempfile
import pytest

from simple_network_sim.mixing_matrix import AgeRange, MixingMatrix, _check_overlap


def test_AgeRange():
    with pytest.raises(Exception) as e_info:
        a = AgeRange("[10,20)")
        assert 10 in a
        assert 15 in a
        assert 20 not in a

        b = AgeRange(10, 20)
        assert a == b
        assert 10 in b
        assert 20 not in b

        c = AgeRange((10,20))
        assert b == c

        d = AgeRange("70+")
        assert 99 in d
        assert 69 not in d

        e = AgeRange("invalid")
    assert e_info.value.args[0] == f'Invalid age range specified: "invalid"'

    with pytest.raises(Exception) as e_info:
        a = AgeRange("[10,5)")
    assert e_info.value.args[0] == f'Invalid age range specified: [10,5)'

    with pytest.raises(Exception) as e_info:
        a = AgeRange(10, 5)
    assert e_info.value.args[0] == f'Invalid age range specified: [10,5)'

    with pytest.raises(Exception) as e_info:
        a = AgeRange(20,30)
        b = AgeRange(25,35)
        _check_overlap(a, b)
    assert e_info.value.args[0] == f"Overlap in age ranges with {a} and {b}"
    with pytest.raises(Exception) as e_info:
        a = AgeRange("70+")
        b = AgeRange("[65,75)")
        _check_overlap(a, b)
    assert e_info.value.args[0] == f"Overlap in age ranges with {a} and {b}"


def test_invalidMixingMatrixFiles():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,15)","[0,15)"',
            '"[0,15)",0.1, 0.05',
            '"[15,30)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Duplicate column header found in mixing matrix"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,15)","[15,30)"',
            '"[0,15)",0.1, 0.05',
            '"[0,15)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Duplicate row header found in mixing matrix"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as fp:
        rows = [
            ',"[0,20)","[15,30)"',
            '"[0,15)",0.1, 0.05',
            '"[15,30)",0.2, 0.2',
        ]
        fp.write("\n".join(rows))
        fp.flush()
        with pytest.raises(Exception) as e_info:
            MixingMatrix(fp.name)
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
            MixingMatrix(fp.name)
        assert e_info.value.args[0] == "Overlap in age ranges with [0,15) and [10,30)"


def test_sampleMixingMatrix(mixing_matrix):
    mm = MixingMatrix(mixing_matrix)
    assert mm[28][57] == 0.097096431
    assert mm["[30,40)"][75] == 0.026352071
    assert mm[(30,40)][75] == 0.026352071
    assert mm[(30,40)][(18,30)] == 0.144896108
