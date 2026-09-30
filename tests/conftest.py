import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.models import Paper


class FakeLLM:
    """Duck-types core.llm.LLMClient without any network calls."""

    def __init__(self, response_fn=None, static_response="Fake response.", provider="groq"):
        self.provider = provider
        self.model = "fake-model"
        self.calls = []
        self._response_fn = response_fn
        self._static = static_response

    def _respond(self, key):
        self.calls.append(key)
        if self._response_fn:
            return self._response_fn(key)
        return self._static

    def complete(self, prompt, system=None):
        return self._respond(prompt)

    def chat(self, messages):
        return self._respond(messages[-1]["content"])

    def stream(self, messages):
        text = self.chat(messages)
        for i in range(0, len(text), 3):
            yield text[i : i + 3]


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def sample_papers():
    return [
        Paper(title="Deep Learning for Vision", abstract="A study of CNNs for images.", authors=["A. Smith"], year=2021, cluster=0),
        Paper(title="Transformers for Vision", abstract="Applying attention to images.", authors=["B. Lee"], year=2022, cluster=0),
        Paper(title="Graph Neural Networks in Chemistry", abstract="GNNs for molecule property prediction.", authors=["C. Wu"], year=2020, cluster=1),
        Paper(title="Reinforcement Learning for Robotics", abstract="RL policies for robot control.", authors=["D. Patel"], year=2023, cluster=1),
    ]
