import os

import git
import pytest

from simple_network_sim.common import Lazy, get_repo_info


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


@pytest.fixture
def git_repo(tmp_path_factory):
    old_cwd = os.getcwd()
    os.chdir(str(tmp_path_factory.mktemp("repo")))
    repo = git.Repo.init()
    repo.create_remote("origin", "http://example.com")
    open("hello", "w").close()
    repo.index.add("hello")
    repo.index.commit("Initial version")
    yield repo
    os.chdir(old_cwd)


@pytest.fixture
def no_git_repo(tmp_path_factory):
    old_cwd = os.getcwd()
    repo = str(tmp_path_factory.mktemp("repo"))
    os.chdir(repo)
    yield repo
    os.chdir(old_cwd)


# pylint: disable=redefined-outer-name
def test_get_repo_info(git_repo):
    info = get_repo_info()
    assert not info.is_dirty
    assert info.git_sha == git_repo.head.commit.hexsha
    assert info.uri == "http://example.com"


# pylint: disable=redefined-outer-name
def test_get_repo_info_is_dirty(git_repo):
    with open("hello", "w") as fp:
        fp.write("hi")

    info = get_repo_info()
    assert info.is_dirty
    assert info.git_sha == git_repo.head.commit.hexsha
    assert info.uri == "http://example.com"


# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
def test_get_repo_info_no_repo(no_git_repo):
    info = get_repo_info()
    assert info.is_dirty
    assert info.git_sha == ""
    assert info.uri == "https://github.com/ScottishCovidResponse/simple_network_sim.git"
