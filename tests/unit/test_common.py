from simple_network_sim.common import Lazy


def test_lazy_never_evaluated():
    evals = []
    Lazy(lambda: evals.append(1))

    assert evals == []


def test_lazy_evaluates_on_str():
    evals = []
    lazy = Lazy(lambda: (evals.append(1), "hi"))

    assert str(lazy) == str((None, "hi"))
    assert evals == [1]


def test_lazy_evaluates_on_repr():
    evals = []
    lazy = Lazy(lambda: (evals.append(1), "hi"))

    assert repr(lazy) == repr((None, "hi"))
    assert evals == [1]
