import torch
import pytest
from src.kv_gpt import MiniGPT


@pytest.fixture
def model():
    torch.manual_seed(42)

    return MiniGPT(
        vocab_size=100,
        no_of_components=6,
        hidden_states=128,
        no_of_heads=8,
        mask=True,
        context_window=32
    ).eval()


@pytest.mark.parametrize("length", [10, 40, 100])
def test_cache_equivalence(model, length):

    x = torch.randint(0, 100, (2, 6))

    without_cache = model.generate(
        x,
        max_new_tokens=length,
        use_cache=False,
        drop_cache=16
    )

    with_cache = model.generate(
        x,
        max_new_tokens=length,
        use_cache=True,
        drop_cache=16
    )

    assert torch.equal(without_cache, with_cache)
    assert with_cache.shape == (2, 6 + length)


@pytest.mark.parametrize("use_cache", [True, False])
def test_context_window(model, use_cache):

    input_lengths = []

    def check_input(module, args):
        input_lengths.append(args[0].size(1))

    hook = model.register_forward_pre_hook(check_input)

    try:
        x = torch.randint(0, 100, (2, 6))

        model.generate(
            x,
            max_new_tokens=100,
            use_cache=use_cache,
            drop_cache=16
        )

        assert len(input_lengths) == 100
        assert max(input_lengths) <= 32

    finally:
        hook.remove()