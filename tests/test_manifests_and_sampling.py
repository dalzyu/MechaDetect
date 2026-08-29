import pandas as pd
import pytest
import torch

from aigc_detector.manifests import (
    assert_forbidden_demonstration_data_absent,
    assert_no_group_leakage,
    assign_splits,
    duplicate_groups,
    is_held_out_generator,
    normalize_generator,
)
from aigc_detector.sampling import EpochWeightedSampler, generator_balanced_weights


def test_generator_holdout_registry_is_explicit_and_stable() -> None:
    assert is_held_out_generator("flux_1_dev")
    assert is_held_out_generator("nano_banana_pro")
    assert is_held_out_generator("novelai_anime_v3")
    assert is_held_out_generator("midjourney")
    assert is_held_out_generator("sdxl_1_0")
    assert is_held_out_generator("dalle2")
    assert not is_held_out_generator("stable_diffusion_1")

def test_duplicate_clustering_links_exact_source_and_near_hashes() -> None:
    groups = duplicate_groups(
        ["a", "b", "c", "c"],
        [0, 1, 2**63, 123456],
        ["", "", "linked", "linked"],
    )
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]


def test_duplicate_group_cannot_cross_splits() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": "a.png",
                "label": "fully_aigc",
                "dataset": "x",
                "generator": "flux",
                "official_split": "train",
                "duplicate_group": "same",
            },
            {
                "image_path": "b.png",
                "label": "authentic",
                "dataset": "x",
                "generator": "",
                "official_split": "validation",
                "duplicate_group": "same",
            },
        ]
    )
    assigned = assign_splits(frame)
    assert assigned["split"].nunique() == 1
    assert_no_group_leakage(assigned)


def test_diffusionforensics_adm_is_forced_into_training_protocol() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": "adm.png",
                "label": "fully_aigc",
                "dataset": "DiffusionForensics",
                "generator": "ADM",
                "official_split": "train",
                "duplicate_group": "adm",
            }
        ]
    )
    assert assign_splits(frame).iloc[0]["split"] == "train"


def test_organizer_demonstration_data_is_rejected_from_training() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "WildFake",
                "generator": "DALL-E",
                "Category": "Advanced",
                "image_path": "dalle/example.png",
                "split": "train",
            }
        ]
    )
    with pytest.raises(ValueError, match="Forbidden organizer demonstration data"):
        assert_forbidden_demonstration_data_absent(frame)


def test_test_unseen_excludes_generator_families_present_in_train() -> None:
    frame = pd.DataFrame(
        [
            {
                "image_path": "train.png",
                "label": "fully_aigc",
                "dataset": "x",
                "generator": "DDIM",
                "official_split": "train",
                "duplicate_group": "train",
            },
            {
                "image_path": "quarantine.png",
                "label": "fully_aigc",
                "dataset": "x",
                "generator": "DDIM",
                "official_split": "test_unseen",
                "duplicate_group": "test",
            },
            {
                "image_path": "vqdm.png",
                "label": "fully_aigc",
                "dataset": "x",
                "generator": "VQDM",
                "official_split": "test_unseen",
                "duplicate_group": "vqdm",
            },
        ]
    )
    assigned = assign_splits(frame)
    splits = dict(zip(assigned["generator_family"], assigned["split"], strict=False))
    assert splits["ddim"] == "test_seen"
    assert splits["vqdm"] == "test_unseen"


def test_sampler_matches_class_and_generator_targets() -> None:
    labels = [0] * 6 + [1] * 2 + [2] * 4
    generators = ["real_a"] * 4 + ["real_b"] * 2 + ["tamper"] * 2 + ["g1"] + ["g2"] * 3
    weights = generator_balanced_weights(labels, generators)
    label_tensor = torch.tensor(labels)
    assert weights[label_tensor == 0].sum().item() == pytest.approx(0.50)
    assert weights[label_tensor == 1].sum().item() == pytest.approx(0.25)
    assert weights[label_tensor == 2].sum().item() == pytest.approx(0.25)
    assert weights[:4].sum().item() == pytest.approx(weights[4:6].sum().item())


def test_epoch_sampler_reproduces_order_without_stored_iterator_state() -> None:
    sampler = EpochWeightedSampler(torch.tensor([0.2, 0.3, 0.5]), samples=20, seed=42)
    sampler.set_epoch(3)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(4)
    assert first != list(sampler)


def test_epoch_sampler_skips_resumed_samples_without_loading_them() -> None:
    sampler = EpochWeightedSampler(torch.tensor([0.2, 0.3, 0.5]), samples=20, seed=42)
    full = list(sampler)
    sampler.set_start_offset(7)
    assert list(sampler) == full[7:]
    assert len(sampler) == 13


def test_distributed_epoch_sampler_gives_every_rank_equal_global_slices() -> None:
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    samplers = [
        EpochWeightedSampler(
            weights, samples=20, seed=42, rank=rank, world_size=6
        )
        for rank in range(6)
    ]

    expected_global = torch.multinomial(
        weights,
        24,  # ceil(20 / 6) * 6
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    ).tolist()
    for rank, sampler in enumerate(samplers):
        assert len(sampler) == 4
        assert list(sampler) == expected_global[rank::6]

    with pytest.raises(ValueError):
        samplers[0].set_start_offset(5)
