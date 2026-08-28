from torch import nn

from aigc_detector.model import ProvenanceModel


def test_common_token_adapter_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(
        "aigc_detector.model.build_backbone",
        lambda *args, **kwargs: nn.Identity(),
    )
    legacy = ProvenanceModel(
        "unused",
        encoder_revision=None,
        encoder_dim=8,
        trunk_dim=4,
        branch_dim=3,
        use_token_adapter=False,
    )
    common = ProvenanceModel(
        "unused",
        encoder_revision=None,
        encoder_dim=8,
        trunk_dim=4,
        branch_dim=3,
        use_token_adapter=True,
    )
    assert isinstance(legacy.token_adapter, nn.Identity)
    assert legacy.heads.aigc_queries.queries.shape[-1] == 8
    assert common.heads.aigc_queries.queries.shape[-1] == 4
