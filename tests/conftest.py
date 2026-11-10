import pytest

from redline.classifier import default_classifier
from redline.pipeline import Reviewer
from redline.playbook import default_playbook


@pytest.fixture(scope="session")
def playbook():
    return default_playbook()


@pytest.fixture(scope="session")
def classifier():
    return default_classifier()


@pytest.fixture(scope="session")
def reviewer(playbook, classifier):
    return Reviewer(playbook=playbook, classifier=classifier)
