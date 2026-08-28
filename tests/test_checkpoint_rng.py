import torch

from aigc_detector.train import cpu_rng_states


def test_checkpoint_rng_states_are_normalized_to_cpu_byte_tensors() -> None:
    states = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
    normalized = cpu_rng_states(states)
    assert len(normalized) == 1
    assert normalized[0].device.type == "cpu"
    assert normalized[0].dtype is torch.uint8
