import asyncio
from types import SimpleNamespace

from phylogenic.benchmark.utils import build_system_prompt
from scripts.run_personality_benchmark import GenomeModel
from scripts.run_ab_benchmark import PhylogenicModel


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.last_messages = None

    async def chat_completion(self, messages, stream=False):
        # capture messages for assertions and yield response
        self.last_messages = messages
        yield self.response


async def _run_generate(model, prompt):
    return await model.generate(prompt)


def test_genomemodel_uses_shared_prompt():
    traits = {"empathy": 0.85}
    genome = SimpleNamespace(traits=traits)
    fake = FakeClient("Answer")
    model = GenomeModel(fake, genome)

    assert model._build_system_prompt() == build_system_prompt(traits)

    # run the async generate
    resp = asyncio.get_event_loop().run_until_complete(_run_generate(model, "Hello"))
    assert resp == "Answer"
    assert fake.last_messages is not None
    assert fake.last_messages[0]["role"] == "system"
    assert build_system_prompt(traits) in fake.last_messages[0]["content"]


def test_phylogenicmodel_uses_shared_prompt():
    traits = {"conciseness": 0.95}
    genome = SimpleNamespace(traits=traits)
    fake = FakeClient("Done")
    model = PhylogenicModel(fake, genome)

    assert model._build_system_prompt() == build_system_prompt(traits)

    resp = asyncio.get_event_loop().run_until_complete(_run_generate(model, "Q"))
    assert resp == "Done"
    assert fake.last_messages is not None
    assert fake.last_messages[0]["role"] == "system"
    assert build_system_prompt(traits) in fake.last_messages[0]["content"]
