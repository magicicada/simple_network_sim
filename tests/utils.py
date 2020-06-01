import os
import tempfile

import pytest
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import remove_ticks_and_titles


def compare_mpl_plots(baseline_name, fig, force_update=False):
    """
    This function compares the image output with a precomputed version. The image name will be `baseline_name`
    suffixed with `.png`. The user of this function is encouraged to name this after the test module and name so that
    there are no collisions.

    This will fail the test if the baseline image is not present, but it will be created and subsequent runs will pass.
    So, if the test fails, remove the baseline image and validate that the newly created image looks like expected.

    :param baseline_name: Name to use for the baseline image. They will all be inside of mpl_images
    :param fig: The produced image you want to check for regressions
    :param force_update: This can be handy during development. Setting this to True will always fail the test, but the
                         baseline image will be updated, so you can easily iterate over the way you want the plotting
                         function called
    """
    baseline = os.path.join(os.path.dirname(__file__), "mpl_images", baseline_name) + ".png"
    remove_ticks_and_titles(fig)
    if not os.path.isfile(baseline) or force_update:
        fig.savefig(baseline)
        pytest.fail("Baseline image not found. We've created it. Running this test again should succeed")
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".png") as fp:
        fig.savefig(fp.name)
        assert compare_images(baseline, fp.name, tol=2) is None, "Images don't match"
