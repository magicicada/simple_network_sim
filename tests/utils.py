import os
import tempfile

import pytest
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import remove_ticks_and_titles


def compare_mpl_plots(fig, force_update=False):
    """
    This function is specialised in comparing matplotlib figures with a baseline. The baseline will be an image inside
    mpl_images in the current directory, with name matching the test's name.

    The baseline will be automatically created if it doesn't exist (but the test will still fail). Using force_update
    allows the user to iterate over how the image looks through running the tests multiple times with the flag enabled.

    :param fig: The produced image you want to check for regressions
    :param force_update: This can be handy during development. Setting this to True will always fail the test, but the
                         baseline image will be updated, so you can easily iterate over the way you want the plotting
                         function called
    """
    test_name = os.environ["PYTEST_CURRENT_TEST"].replace(" (call)", "").replace("::", "__").replace(".py", "")
    test_dir = os.path.dirname(test_name)
    test_name = os.path.basename(test_name)
    baseline = os.path.join(test_dir, "mpl_images", test_name) + ".png"
    remove_ticks_and_titles(fig)
    if not os.path.isfile(baseline) or force_update:
        fig.savefig(baseline)
        pytest.fail("Baseline image not found. We've created it. Running this test again should succeed")
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".png", delete=False) as fp:
        fig.savefig(fp.name)
        assert compare_images(baseline, fp.name, tol=2) is None, "Images don't match"


def create_baseline(test_data, force_update=False):
    """
    This function is a generic way to keep a baseline fixture updated. It's specially useful for regression tests. It
    checks if the baseline_name exists in the desired path, if it doesn't the test will fail, but this function will
    write it (so next run will succeed).

    This function doesn't make any assumptions regarding the data inside the files, so it just returns the baseline so
    that the test can load it and compare.

    :param test_data: the file produced by the test
    :param force_update: set it to true if you want to always overwrite the baseline file (the test will always fail)
    """
    test_name = os.environ["PYTEST_CURRENT_TEST"].replace(" (call)", "").replace("::", "__")
    test_dir = os.path.dirname(test_name)
    test_name = os.path.basename(test_name)
    baseline = os.path.join(test_dir, "data", test_name)
    if not os.path.isfile(baseline) or force_update:
        try:
            os.unlink(baseline)
        except FileNotFoundError:
            pass
        os.link(test_data, baseline)
        pytest.fail("Baseline image not found. We've created it. Running this test again should succeed")
    return baseline


def calculateInfectiousOverTime(ts, infectiousStates):
    """
    Create a list of the number of infectious people over time

    :param ts: pandas dataframe with the entire outbreak timeseries
    :param infectiousStates: compartments considered infectious
    """
    return ts[ts.state.isin(infectiousStates)].groupby("time").sum().total.to_list()
