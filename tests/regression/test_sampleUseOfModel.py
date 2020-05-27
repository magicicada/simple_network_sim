import hashlib
import os
import tempfile

from simple_network_sim import sampleUseOfModel


def test_run_seeded(initial_infection):
    with tempfile.NamedTemporaryFile(mode="wb+", delete=False, suffix=".png") as fp:
        sampleUseOfModel.main(["seeded", initial_infection, fp.name])

        # TODO: improve this once we output a different kind of format
        fp.seek(0)
        checksum = hashlib.md5()
        checksum.update(fp.read())
        assert checksum.hexdigest() == "697ce151437d64264a9a9ea8edb37049"
