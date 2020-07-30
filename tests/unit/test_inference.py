import datetime as dt
from unittest.mock import patch, call

import pytest

import numpy as np
import pandas as pd

from simple_network_sim import inference as inf


def test_uniform_pdf():
    assert inf.uniform_pdf(0.5, 0., 1.) == 1.
    assert inf.uniform_pdf(0., 0., 1.) == 1.
    assert inf.uniform_pdf(1., 0., 1.) == 1.
    assert inf.uniform_pdf(1.5, 0., 1.) == 0.
    assert inf.uniform_pdf(1.5, 0., 2.) == 0.5


def test_lognormal():
    lognormal = inf.lognormal(10., 0.1)
    assert lognormal.pdf(10.) == 0.3994396139930765
    assert lognormal.pdf(0.) == 0.
    assert np.allclose(lognormal.stats(), (np.array(10.), np.array(1.)))

    lognormal = inf.lognormal(10., 0.1, 5.)
    assert lognormal.pdf(10.) == 0.0821304389446951
    assert lognormal.pdf(0.) == 0.
    assert np.allclose(lognormal.stats(), (np.array(10.), np.array(25.)))


def test_split_dataframe():
    df = pd.DataFrame([
        {"Date": "2020-01-01", "Contact_Multiplier": 1.},
        {"Date": "2020-02-01", "Contact_Multiplier": 2.},
        {"Date": "2020-03-01", "Contact_Multiplier": 3.},
        {"Date": "2020-04-01", "Contact_Multiplier": 4.},
        {"Date": "2020-05-01", "Contact_Multiplier": 5.},
    ])
    partitions = [dt.date.min, dt.date(2020, 1, 17), dt.date(2020, 2, 28), dt.date(9999, 12, 31)]

    values = list(inf.split_dataframe(df, partitions))
    assert values[0][0] == 1.
    pd.testing.assert_series_equal(values[0][1], pd.Series([True, False, False, False, False], name="Date"))

    assert values[1][0] == 2.
    pd.testing.assert_series_equal(values[1][1], pd.Series([False, True, False, False, False], name="Date"))

    assert values[2][0] == 3.
    pd.testing.assert_series_equal(values[2][1], pd.Series([False, False, True, True, True], name="Date"))


def test_split_dataframe_non_overlapping():
    df = pd.DataFrame([
        {"Date": "2020-01-01", "Contact_Multiplier": 1.},
        {"Date": "2020-02-01", "Contact_Multiplier": 2.},
        {"Date": "2020-03-01", "Contact_Multiplier": 3.},
        {"Date": "2020-04-01", "Contact_Multiplier": 4.},
        {"Date": "2020-05-01", "Contact_Multiplier": 5.},
    ])
    partitions = [dt.date.min, dt.date(2020, 12, 17), dt.date(9999, 12, 31)]

    values = list(inf.split_dataframe(df, partitions))
    assert len(values) == 1
    assert values[0][0] == 1.
    pd.testing.assert_series_equal(values[0][1], pd.Series([True, True, True, True, True], name="Date"))


def test_split_dataframe_no_dates():
    df = pd.DataFrame([
        {"Date": "2020-01-01", "Contact_Multiplier": 1.},
        {"Date": "2020-02-01", "Contact_Multiplier": 2.},
        {"Date": "2020-03-01", "Contact_Multiplier": 3.},
        {"Date": "2020-04-01", "Contact_Multiplier": 4.},
        {"Date": "2020-05-01", "Contact_Multiplier": 5.},
    ])
    partitions = [dt.date.min, dt.date(9999, 12, 31)]

    values = list(inf.split_dataframe(df, partitions))
    assert len(values) == 1
    assert values[0][0] == 1.
    pd.testing.assert_series_equal(values[0][1], pd.Series([True, True, True, True, True], name="Date"))


def test_InferredInfectionProbability():
    ip = inf.InferredInfectionProbability(pd.DataFrame(), pd.DataFrame(), 2.0, 1.0, np.random.default_rng(123))
    assert ip.value.empty
    assert ip.mean.empty
    assert ip.shape == 2.
    assert ip.kernel_sigma == 1.
    assert ip.rng.__eq__(np.random.default_rng(123))


def test_InferredInfectionProbability_generate_from_prior(abcsmc):
    ip = inf.InferredInfectionProbability.generate_from_prior(abcsmc)
    assert ip.kernel_sigma == 0.1
    assert ip.shape == 4.
    pd.testing.assert_frame_equal(ip.mean, pd.DataFrame([{"Date": "2020-01-01", "Value": 0.5}]))
    pd.testing.assert_frame_equal(ip.value, pd.DataFrame([{"Date": "2020-01-01", "Value": 0.236177258450}]))
    assert ip.rng.__eq__(abcsmc.rng)


def test_InferredInfectionProbability_generate_perturbated():
    value = pd.DataFrame([{"Time": 0., "Value": 0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 1.}])
    ip1 = inf.InferredInfectionProbability(value, mean, 2.0, 1.0, np.random.default_rng(123))
    ip2 = ip1.generate_perturbated()

    pd.testing.assert_frame_equal(ip1.value, value)
    pd.testing.assert_frame_equal(ip2.value, pd.DataFrame([{"Time": 0., "Value": 0.682351863248}]))
    pd.testing.assert_frame_equal(ip1.mean, ip2.mean)
    assert ip2.shape == ip1.shape
    assert ip2.kernel_sigma == ip1.kernel_sigma
    assert ip2.rng.__eq__(ip2.rng)


def test_InferredInfectionProbability_generate_perturbated_multiple_values():
    value = pd.DataFrame([{"Time": 0., "Value": 0.5}, {"Time": 0., "Value": 0.75}])
    mean = pd.DataFrame([{"Time": 0., "Value": 1.}, {"Time": 0., "Value": 1.5}])
    ip1 = inf.InferredInfectionProbability(value, mean, 2.0, 1.0, np.random.default_rng(123))
    ip2 = ip1.generate_perturbated()

    pd.testing.assert_frame_equal(ip1.value, value)
    pd.testing.assert_frame_equal(ip2.value, pd.DataFrame([{"Time": 0., "Value": 0.682351863248},
                                                           {"Time": 0., "Value": 0.053821018802}]))
    pd.testing.assert_frame_equal(ip1.mean, ip2.mean)
    assert ip2.shape == ip1.shape
    assert ip2.kernel_sigma == ip1.kernel_sigma
    assert ip2.rng.__eq__(ip2.rng)


def test_InferredInfectionProbability_validate():
    value = pd.DataFrame([{"Time": 0., "Value": 0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 1.}])
    ip1 = inf.InferredInfectionProbability(value, mean, 2.0, 1.0, np.random.default_rng(123))
    assert ip1.validate()

    value = pd.DataFrame([{"Time": 0., "Value": -0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 1.}])
    ip1 = inf.InferredInfectionProbability(value, mean, 2.0, 1.0, np.random.default_rng(123))
    assert ~ip1.validate()


@pytest.mark.parametrize("proba", [0.5, 0])
def test_InferredInfectionProbability_prior_pdf_beta_degenerated_to_uniform(proba):
    value = pd.DataFrame([{"Time": 0., "Value": proba}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.5}])
    ip1 = inf.InferredInfectionProbability(value, mean, 1.0, 1.0, np.random.default_rng(123))
    assert ip1.prior_pdf() == 1.


@pytest.mark.parametrize("proba", [-1., 1.5])
def test_InferredInfectionProbability_prior_pdf_beta_degenerated_to_uniform_zero_pdf(proba):
    value = pd.DataFrame([{"Time": 0., "Value": proba}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.5}])
    ip1 = inf.InferredInfectionProbability(value, mean, 1.0, 1.0, np.random.default_rng(123))
    assert ip1.prior_pdf() == 0.


@pytest.mark.parametrize("proba", [0.5, 0.])
def test_InferredInfectionProbability_prior_pdf_beta_degenerated_to_uniform_several_values(proba):
    value = pd.DataFrame([{"Time": 0., "Value": proba}, {"Time": 1., "Value": 0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.5}, {"Time": 1., "Value": 0.5}])
    ip1 = inf.InferredInfectionProbability(value, mean, 1.0, 1.0, np.random.default_rng(123))
    assert ip1.prior_pdf() == 1.

    value = pd.DataFrame([{"Time": 0., "Value": proba}, {"Time": 1., "Value": 0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.5}, {"Time": 1., "Value": 0.5}])
    ip1 = inf.InferredInfectionProbability(value, mean, 1.0, 1.0, np.random.default_rng(123))
    assert ip1.prior_pdf() == 1.


@pytest.mark.parametrize("proba", [-1., 1.5])
def test_InferredInfectionProbability_prior_pdf_beta_degenerated_to_uniform_several_values_zero_pdf(proba):
    value = pd.DataFrame([{"Time": 0., "Value": proba}, {"Time": 1., "Value": 0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.5}, {"Time": 1., "Value": 0.5}])
    ip1 = inf.InferredInfectionProbability(value, mean, 1.0, 1.0, np.random.default_rng(123))
    assert ip1.prior_pdf() == 0.


def test_InferredInfectionProbability_prior_pdf_beta():
    value = pd.DataFrame([{"Time": 0., "Value": 0.05}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.05}])
    ip1 = inf.InferredInfectionProbability(value, mean, 4.0, 1.0, np.random.default_rng(123))
    assert pytest.approx(ip1.prior_pdf(), 16.03449072358245, 1e-8)


def test_InferredInfectionProbability_prior_pdf_beta_several_values():
    value = pd.DataFrame([{"Time": 0., "Value": 0.05}, {"Time": 1., "Value": 0.05}])
    mean = pd.DataFrame([{"Time": 0., "Value": 0.05}, {"Time": 1., "Value": 0.05}])
    ip1 = inf.InferredInfectionProbability(value, mean, 4.0, 1.0, np.random.default_rng(123))
    assert pytest.approx(ip1.prior_pdf(), 16.03449072358245**2, 1e-8)


def test_InferredInfectionProbability_perturbation_pdf_unit_noise():
    value = pd.DataFrame([{"Time": 0., "Value": 0.05}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInfectionProbability(value, mean, 4.0, 1.0, np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.05}])) == 1.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.}])) == 1.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.}])) == 1.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": -0.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 2.05}])) == 0.


def test_InferredInfectionProbability_perturbation_pdf_small_noise():
    value = pd.DataFrame([{"Time": 0., "Value": 0.05}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInfectionProbability(value, mean, 4.0, 0.01, np.random.default_rng(123))

    pytest.approx(ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.05}])), 50., 1e-8)
    pytest.approx(ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.06}])), 50., 1e-8)
    pytest.approx(ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.04}])), 50., 1e-8)
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.07}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.03}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": -0.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 2.05}])) == 0.


def test_InferredInfectionProbability_perturbation_pdf_multiple_values_unit_noise():
    value = pd.DataFrame([{"Time": 0., "Value": 0.05}, {"Time": 1., "Value": 0.05}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInfectionProbability(value, mean, 4.0, 1.0, np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.05}, {"Time": 1., "Value": 0.05}])) == 1.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.}, {"Time": 1., "Value": 0.}])) == 1.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.}, {"Time": 1., "Value": 1.}])) == 1.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": -0.05}, {"Time": 1., "Value": -0.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.05}, {"Time": 1., "Value": 1.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 2.05}, {"Time": 1., "Value": 2.05}])) == 0.


def test_InferredInfectionProbability_perturbation_pdf_multiple_values_small_noise():
    value = pd.DataFrame([{"Time": 0., "Value": 0.05}, {"Time": 1., "Value": 0.05}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInfectionProbability(value, mean, 4.0, 0.01, np.random.default_rng(123))

    pytest.approx(ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.05},
                                                     {"Time": 0., "Value": 0.05}])), 50.**2, 1e-8)
    pytest.approx(ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.06},
                                                     {"Time": 0., "Value": 0.06}])), 50.**2, 1e-8)
    pytest.approx(ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.04},
                                                     {"Time": 0., "Value": 0.04}])), 50.**2, 1e-8)
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.07}, {"Time": 0., "Value": 0.07}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.03}, {"Time": 0., "Value": 0.03}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 0.}, {"Time": 0., "Value": 0.}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.}, {"Time": 0., "Value": 1.}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": -0.05}, {"Time": 0., "Value": -0.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 1.05}, {"Time": 0., "Value": 1.05}])) == 0.
    assert ip1.perturbation_pdf(pd.DataFrame([{"Time": 0., "Value": 2.05}, {"Time": 0., "Value": 2.05}])) == 0.


def test_InferredInitialInfections():
    ip = inf.InferredInitialInfections(pd.DataFrame(), pd.DataFrame(), 0.2, 10., 1., np.random.default_rng(123))
    assert ip.value.empty
    assert ip.mean.empty
    assert ip.stddev == 0.2
    assert ip.stddev_min == 10.
    assert ip.kernel_sigma == 1.
    assert ip.rng.__eq__(np.random.default_rng(123))


def test_InferredInitialInfections_generate_from_prior(data_api, abcsmc):
    ip = inf.InferredInitialInfections.generate_from_prior(abcsmc)
    assert ip.kernel_sigma == 10.
    assert ip.stddev == .2
    assert ip.stddev_min == 10.
    pd.testing.assert_frame_equal(ip.mean, data_api.read_table("human/initial-infections", "initial-infections"))
    pd.testing.assert_frame_equal(ip.value, pd.DataFrame([
        {"Health_Board": "S08000016", "Age": "[17,70)", "Infected": 80.61398}]))
    assert ip.rng.__eq__(abcsmc.rng)


def test_InferredInitialInfections_generate_perturbated():
    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 15},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))
    ip2 = ip1.generate_perturbated()

    pd.testing.assert_frame_equal(ip1.value, value)
    pd.testing.assert_frame_equal(ip2.value, pd.DataFrame([
        {"Health_Board": "hb1", "Age": "[17,70)", "Infected": 15.364703826496},
        {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 29.107642037604}]))
    pd.testing.assert_frame_equal(ip1.mean, ip2.mean)
    assert ip2.stddev == ip1.stddev
    assert ip2.stddev_min == ip1.stddev_min
    assert ip2.kernel_sigma == ip1.kernel_sigma
    assert ip2.rng.__eq__(ip2.rng)


def test_InferredInitialInfections_validate():
    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 0},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))
    assert ip1.validate()

    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": -10},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 10}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))
    assert ~ip1.validate()


def test_InferredInitialInfections_prior_pdf():
    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 0},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))
    assert ip1.prior_pdf() == 0.

    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))
    assert pytest.approx(ip1.prior_pdf(), 0.00013613607439537107, 1e-8)


@pytest.mark.parametrize("bump", [-1., -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.])
def test_InferredInitialInfections_perturbation_pdf_unit_noise(bump):
    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([
        {"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10 + bump},
        {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30 + bump}])) == 0.25


@pytest.mark.parametrize("bump", [-1.01, -5, 1.01, 5, 10])
def test_InferredInitialInfections_perturbation_pdf_unit_noise_too_far(bump):
    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([
        {"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10 + bump},
        {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30 + bump}])) == 0.


@pytest.mark.parametrize("bump", [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.])
def test_InferredInitialInfections_perturbation_pdf_value_close_to_zero(bump):
    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 0.5},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame()
    ip1 = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([
        {"Health_Board": "hb1", "Age": "[17,70)", "Infected": 0.5 + bump},
        {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30 + bump}])) == (1. / 1.5) * 0.5


def test_InferredContactMultipliers():
    ip = inf.InferredContactMultipliers(pd.DataFrame(), pd.DataFrame(), 0.2, 1., [], np.random.default_rng(123))
    assert ip.value.empty
    assert ip.mean.empty
    assert ip.stddev == 0.2
    assert ip.kernel_sigma == 1.
    assert ip.partitions == []
    assert ip.rng.__eq__(np.random.default_rng(123))


def test_InferredContactMultipliers_generate_from_prior(data_api, abcsmc):
    ip = inf.InferredContactMultipliers.generate_from_prior(abcsmc)
    assert ip.kernel_sigma == 0.2
    assert ip.stddev == .2
    assert ip.partitions == [dt.date.min, dt.date(2020, 3, 24), dt.date(2020, 4, 3), dt.date(9999, 12, 31)]
    pd.testing.assert_frame_equal(ip.mean, data_api.read_table("human/movement-multipliers", "movement-multipliers"))
    pd.testing.assert_series_equal(ip.value.Contact_Multiplier,
                                   pd.Series([0.806140, 0.045585, 0.045585, 0.045585, 0.045585],
                                             name="Contact_Multiplier"))
    assert ip.rng.__eq__(abcsmc.rng)


def test_InferredContactMultipliers_generate_perturbated():
    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 4.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    ip2 = ip1.generate_perturbated()

    pd.testing.assert_frame_equal(ip1.value, value)
    pd.testing.assert_frame_equal(ip2.value, pd.DataFrame([
        {"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 4.364703726496},
        {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.107642037604}]))
    pd.testing.assert_frame_equal(ip1.mean, ip2.mean)
    assert ip2.stddev == ip1.stddev
    assert ip2.kernel_sigma == ip1.kernel_sigma
    assert ip2.partitions == ip1.partitions
    assert ip2.rng.__eq__(ip2.rng)


def test_InferredContactMultipliers_validate():
    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 4.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    assert ip1.validate()

    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 0.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    assert ~ip1.validate()

    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": -4.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    assert ~ip1.validate()


def test_InferredContactMultipliers_prior_pdf():
    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 0.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    assert ip1.prior_pdf() == 0.

    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 10000.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    assert pytest.approx(ip1.prior_pdf(), 0.0, 1e-8)

    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 3.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))
    assert pytest.approx(ip1.prior_pdf(), 0.012751291705831346, 1e-8)


@pytest.mark.parametrize("bump", [-1., -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.])
def test_InferredContactMultipliers_perturbation_pdf_unit_noise(bump):
    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 3.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame()
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([
        {"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 3. + bump},
        {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5. + bump}])) == 0.25


@pytest.mark.parametrize("bump", [-1.01, -5, 1.01, 5, 10])
def test_InferredContactMultipliers_perturbation_pdf_unit_noise_too_far(bump):
    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 3.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame()
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([
        {"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 3. + bump},
        {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5. + bump}])) == 0.


@pytest.mark.parametrize("bump", [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.])
def test_InferredContactMultipliers_perturbation_pdf_value_close_to_zero(bump):
    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 0.5},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame()
    ip1 = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max],
                                         np.random.default_rng(123))

    assert ip1.perturbation_pdf(pd.DataFrame([
        {"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 0.5 + bump},
        {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5. + bump}])) == (1. / 1.5) * 0.5


def test_create_abcsmc(data_api):
    inf.ABCSMC(
        data_api.read_table("human/abcsmc-parameters", "abcsmc-parameters"),
        data_api.read_table("human/historical-deaths", "historical-deaths"),
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population"),
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        data_api.read_table("human/infection-probability", "infection-probability"),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/start-end-date", "start-end-date"),
        data_api.read_table("human/movement-multipliers", "movement-multipliers"),
        data_api.read_table("human/stochastic-mode", "stochastic-mode"),
        data_api.read_table("human/random-seed", "random-seed"),
    )


@pytest.mark.parametrize("param_index", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_create_abcsmc_invalid_abc_smc_params(data_api, param_index):
    parameters = data_api.read_table("human/abcsmc-parameters", "abcsmc-parameters")
    parameters.at[param_index, "Value"] = 0

    with pytest.raises(AssertionError):
        inf.ABCSMC(
            parameters,
            data_api.read_table("human/historical-deaths", "historical-deaths"),
            data_api.read_table("human/compartment-transition", "compartment-transition"),
            data_api.read_table("human/population", "population"),
            data_api.read_table("human/commutes", "commutes"),
            data_api.read_table("human/mixing-matrix", "mixing-matrix"),
            data_api.read_table("human/infection-probability", "infection-probability"),
            data_api.read_table("human/initial-infections", "initial-infections"),
            data_api.read_table("human/infectious-compartments", "infectious-compartments"),
            data_api.read_table("human/trials", "trials"),
            data_api.read_table("human/start-end-date", "start-end-date"),
            data_api.read_table("human/movement-multipliers", "movement-multipliers"),
            data_api.read_table("human/stochastic-mode", "stochastic-mode"),
            data_api.read_table("human/random-seed", "random-seed"),
        )


def test_create_abcsmc_too_many_trials(data_api):
    with pytest.raises(AssertionError):
        inf.ABCSMC(
            data_api.read_table("human/abcsmc-parameters", "abcsmc-parameters"),
            data_api.read_table("human/historical-deaths", "historical-deaths"),
            data_api.read_table("human/compartment-transition", "compartment-transition"),
            data_api.read_table("human/population", "population"),
            data_api.read_table("human/commutes", "commutes"),
            data_api.read_table("human/mixing-matrix", "mixing-matrix"),
            data_api.read_table("human/infection-probability", "infection-probability"),
            data_api.read_table("human/initial-infections", "initial-infections"),
            data_api.read_table("human/infectious-compartments", "infectious-compartments"),
            pd.DataFrame([{"Value": 2}]),
            data_api.read_table("human/start-end-date", "start-end-date"),
            data_api.read_table("human/movement-multipliers", "movement-multipliers"),
            data_api.read_table("human/stochastic-mode", "stochastic-mode"),
            data_api.read_table("human/random-seed", "random-seed"),
        )


def test_create_abcsmc_too_many_infection_probas(data_api):
    with pytest.raises(AssertionError):
        inf.ABCSMC(
            data_api.read_table("human/abcsmc-parameters", "abcsmc-parameters"),
            data_api.read_table("human/historical-deaths", "historical-deaths"),
            data_api.read_table("human/compartment-transition", "compartment-transition"),
            data_api.read_table("human/population", "population"),
            data_api.read_table("human/commutes", "commutes"),
            data_api.read_table("human/mixing-matrix", "mixing-matrix"),
            pd.DataFrame([{"Date": "2020-01-01", "Value": 0.5}, {"Date": "2020-01-02", "Value": 0.5}]),
            data_api.read_table("human/initial-infections", "initial-infections"),
            data_api.read_table("human/infectious-compartments", "infectious-compartments"),
            data_api.read_table("human/trials", "trials"),
            data_api.read_table("human/start-end-date", "start-end-date"),
            data_api.read_table("human/movement-multipliers", "movement-multipliers"),
            data_api.read_table("human/stochastic-mode", "stochastic-mode"),
            data_api.read_table("human/random-seed", "random-seed"),
        )


def test_Particle():
    rng = np.random.default_rng(123)
    infection_probability = inf.InferredInfectionProbability(pd.DataFrame(), pd.DataFrame(), 2.0, 1.0, rng)
    initial_infections = inf.InferredInitialInfections(pd.DataFrame(), pd.DataFrame(), 0.2, 10., 1., rng)
    contact_multipliers = inf.InferredContactMultipliers(pd.DataFrame(), pd.DataFrame(), 0.2, 1., [], rng)

    particle = inf.Particle({
        "infection-probability": infection_probability,
        "initial-infections": initial_infections,
        "contact-multipliers": contact_multipliers
    })

    assert len(particle.inferred_variables) == 3
    assert list(particle.inferred_variables.keys()) == ["infection-probability", "initial-infections",
                                                        "contact-multipliers"]

    inf.Particle({})


def test_Particle_generate_from_priors(abcsmc):
    particle = inf.Particle.generate_from_priors(abcsmc)
    assert list(particle.inferred_variables.keys()) == ["infection-probability", "initial-infections",
                                                        "contact-multipliers"]
    pd.testing.assert_frame_equal(particle.inferred_variables["infection-probability"].value,
                                  pd.DataFrame([{"Date": "2020-01-01", "Value": 0.23617725844}]))
    pd.testing.assert_frame_equal(particle.inferred_variables["initial-infections"].value,
                                  pd.DataFrame([{"Health_Board": "S08000016", "Age": "[17,70)", "Infected": 117.66062}])
                                  )
    pd.testing.assert_frame_equal(particle.inferred_variables["contact-multipliers"].value.head(1),
                                  pd.DataFrame([{"Date": "2020-03-16", "Movement_Multiplier": 1.,
                                                 "Contact_Multiplier": 1.09931}]))


def test_Particle_generate_perturbated(abcsmc):
    particle = inf.Particle.generate_from_priors(abcsmc)
    particle2 = particle.generate_perturbated()

    for key in ["infection-probability", "initial-infections", "contact-multipliers"]:
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(particle.inferred_variables[key].value,
                                          particle2.inferred_variables[key].value)

        pd.testing.assert_frame_equal(particle.inferred_variables[key].mean, particle2.inferred_variables[key].mean)

    assert particle2.inferred_variables["infection-probability"].value.Value.values[0] == 0.19149213800914885
    assert particle2.inferred_variables["initial-infections"].value.Infected.values[0] == 124.055711566246
    assert particle2.inferred_variables["contact-multipliers"].value.Contact_Multiplier.values[0] == 1.2552647003896826


def test_Particle_validate():
    rng = np.random.default_rng(123)

    value = pd.DataFrame([{"Time": 0., "Value": 0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 1.}])
    good_infection_proba = inf.InferredInfectionProbability(value, mean, 2.0, 1.0, rng)

    value = pd.DataFrame([{"Time": 0., "Value": -0.5}])
    mean = pd.DataFrame([{"Time": 0., "Value": 1.}])
    bad_infection_proba = inf.InferredInfectionProbability(value, mean, 2.0, 1.0, rng)

    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 0},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 30}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    good_initial_infections = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., rng)

    value = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": -10},
                          {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 10}])
    mean = pd.DataFrame([{"Health_Board": "hb1", "Age": "[17,70)", "Infected": 10},
                         {"Health_Board": "hb2", "Age": "[17,70)", "Infected": 50}])
    bad_initial_infections = inf.InferredInitialInfections(value, mean, 0.2, 10., 1., rng)

    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 4.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    good_cm = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max], rng)

    value = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 0.},
                          {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 5.}])
    mean = pd.DataFrame([{"Date": "2020-01-10", "Movement_Multiplier": 1., "Contact_Multiplier": 2.},
                         {"Date": "2020-02-10", "Movement_Multiplier": 2., "Contact_Multiplier": 4.}])
    bad_cm = inf.InferredContactMultipliers(value, mean, 0.2, 1., [dt.date.min, dt.date(2020, 2, 1), dt.date.max], rng)

    assert inf.Particle({
        "infection-probability": good_infection_proba,
        "initial-infections": good_initial_infections,
        "contact-multipliers": good_cm
    }).validate_particle()

    assert ~inf.Particle({
        "infection-probability": bad_infection_proba,
        "initial-infections": bad_initial_infections,
        "contact-multipliers": bad_cm
    }).validate_particle()


def test_Particle_resample_and_perturbate(abcsmc):
    rng = np.random.default_rng(123)

    particles = [
        inf.Particle.generate_from_priors(abcsmc),
        inf.Particle.generate_from_priors(abcsmc),
    ]

    particle = inf.Particle.resample_and_perturbate(particles, [0.5, 0.5], rng)

    assert particle.inferred_variables["infection-probability"].value.Value.values[0] == 0.6574409275433721
    assert particle.inferred_variables["initial-infections"].value.Infected.values[0] == 75.0824682121024
    assert particle.inferred_variables["contact-multipliers"].value.Contact_Multiplier.values[0] == 1.41267213365044


def test_Particle_prior_pdf(abcsmc):
    particle = inf.Particle.generate_from_priors(abcsmc)
    assert particle.prior_pdf() == 0.5440577875971164


def test_Particle_perturbation_pdf(abcsmc):
    particle = inf.Particle.generate_from_priors(abcsmc)
    particle2 = particle.generate_perturbated()
    assert particle2.perturbation_pdf(particle) == 1.9244621636280501
    assert particle2.perturbation_pdf(particle2) == 1.9244621636280501
    assert particle.perturbation_pdf(particle) == 2.569661390086853


def test_ABCSMC_update_threshold(abcsmc):
    abcsmc.update_threshold([1.0, 1.0, 1.3, 0.9, 1.0])
    assert abcsmc.threshold == 1.


def test_ABCSMC_run_model(abcsmc):
    particle = inf.Particle.generate_from_priors(abcsmc)
    result = abcsmc.run_model(particle)
    assert result[result.state == "D"].total.sum() == 1190.9201532330842


def test_ABCSMC_compute_distance(abcsmc):
    particle = inf.Particle.generate_from_priors(abcsmc)
    result = abcsmc.run_model(particle)
    diff = abcsmc.compute_distance(result)
    assert diff == 21.218467509977117


def test_ABCSMC_compute_weights(abcsmc):
    particles = [
        inf.Particle.generate_from_priors(abcsmc),
        inf.Particle.generate_from_priors(abcsmc),
        inf.Particle.generate_from_priors(abcsmc),
    ]

    particle = particles[0].generate_perturbated()

    assert inf.ABCSMC.compute_weight(0, particles, [1., 1., 1.], particle) == 1.
    assert inf.ABCSMC.compute_weight(1, particles, [1., 1., 1.], particle) == 0.09444160575494241


@patch("simple_network_sim.inference.logger.warning")
def test_ABCSMC_run_model_with_issues(mock_warn_logger, data_api):
    abcsmc = inf.ABCSMC(
        data_api.read_table("human/abcsmc-parameters", "abcsmc-parameters"),
        data_api.read_table("human/historical-deaths", "historical-deaths"),
        data_api.read_table("human/compartment-transition", "compartment-transition"),
        data_api.read_table("human/population", "population").iloc[:10],
        data_api.read_table("human/commutes", "commutes"),
        data_api.read_table("human/mixing-matrix", "mixing-matrix"),
        pd.DataFrame([{"Date": "2020-01-01", "Value": 0.5}]),
        data_api.read_table("human/initial-infections", "initial-infections"),
        data_api.read_table("human/infectious-compartments", "infectious-compartments"),
        data_api.read_table("human/trials", "trials"),
        data_api.read_table("human/start-end-date", "start-end-date"),
        data_api.read_table("human/movement-multipliers", "movement-multipliers"),
        data_api.read_table("human/stochastic-mode", "stochastic-mode"),
        data_api.read_table("human/random-seed", "random-seed"),
    )

    particle = inf.Particle.generate_from_priors(abcsmc)
    abcsmc.run_model(particle)

    mock_warn_logger.assert_has_calls([call("We had %s issues when running the model:", 36)])
