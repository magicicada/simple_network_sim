import pytest

from simple_network_sim.mixing_matrix import AgeRange, MixingMatrix


def test_AgeRange():
    with pytest.raises(Exception) as e_info:
        a = AgeRange("[10,20)")
        assert a.contains(10)
        assert a.contains(15)
        assert not a.contains(20)

        b = AgeRange(10, 20)
        assert a == b
        assert b.contains(10)
        assert not b.contains(20)

        c = AgeRange((10,20))
        assert b == c

        d = AgeRange("70+")
        assert d.contains(99)
        assert not d.contains(69)

        e = AgeRange("invalid")
    assert e_info.value.args[0] == f'Invalid age range specified: "invalid"'

def test_sampleMixingMatrix(mixing_matrix):
    mm = MixingMatrix(mixing_matrix)
    assert mm[28][57] == 0.097096431
    assert mm["[30,40)"][75] == 0.026352071
    assert mm[(30,40)][75] == 0.026352071
    assert mm[(30,40)][(18,30)] == 0.144896108
