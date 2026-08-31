import torch

from finetune_esm.data.batching import EmbeddingBatch
from finetune_esm.models import convbert, fusion, heads

IN_DIM, HIDDEN = 32, 16


def _batch(batch_size=3, ligand_len=7, receptor_len=5, pad_fill=0.0, chains=1):
    torch.manual_seed(0)
    lm = torch.ones(batch_size, ligand_len, dtype=torch.bool)
    lm[0, -2:] = False
    rm = torch.ones(batch_size, receptor_len, dtype=torch.bool)
    lid = torch.zeros(batch_size, ligand_len, dtype=torch.long)
    if chains > 1:
        lid[:, ligand_len // 2 :] = 1
    lid[~lm] = -1
    rid = torch.zeros(batch_size, receptor_len, dtype=torch.long)
    lig = torch.randn(batch_size, ligand_len, IN_DIM)
    rec = torch.randn(batch_size, receptor_len, IN_DIM)
    lig[~lm] = pad_fill
    return EmbeddingBatch(
        lig, lm, lid, rec, rm, rid, torch.zeros(batch_size), torch.arange(batch_size)
    )


def test_encoder_projects_to_hidden_dim_and_reports_it():
    enc = convbert.ConvBertResidueEncoder(IN_DIM, hidden_dim=HIDDEN)
    out = enc(torch.randn(2, 9, IN_DIM), torch.ones(2, 9, dtype=torch.bool))
    assert out.shape == (2, 9, HIDDEN)
    assert enc.out_dim == HIDDEN


def test_encoder_zeroes_padded_positions():
    enc = convbert.ConvBertResidueEncoder(IN_DIM, hidden_dim=HIDDEN).eval()
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4:] = False
    with torch.no_grad():
        out = enc(torch.randn(1, 6, IN_DIM), mask)
    assert torch.all(out[0, 4:] == 0)


def test_encoder_output_is_invariant_to_garbage_in_padding():
    """The additive mask must exclude padding, not merely down-weight it."""
    enc = convbert.ConvBertResidueEncoder(IN_DIM, hidden_dim=HIDDEN).eval()
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4:] = False
    torch.manual_seed(1)
    x = torch.randn(1, 6, IN_DIM)
    dirty = x.clone()
    dirty[0, 4:] = 1e4
    with torch.no_grad():
        assert torch.allclose(enc(x, mask), enc(dirty, mask), atol=1e-4)


def test_fusion_out_dim_matches_the_inner_fusion_at_hidden_dim():
    for inner in ("ec", "sc", "hp", "pad"):
        f = convbert.ConvBertFusion(IN_DIM, inner=inner, hidden_dim=HIDDEN)
        expected = fusion.FUSIONS[inner](HIDDEN).out_dim
        assert f.out_dim == expected
        assert f(_batch()).shape == (3, expected)


def test_fusion_composes_with_the_prediction_head():
    f = convbert.ConvBertFusion(IN_DIM, inner="hp", hidden_dim=HIDDEN)
    head = heads.MLPHead(f.out_dim)
    assert head(f(_batch())).shape == (3,)


def test_fusion_is_receptor_sensitive():
    torch.manual_seed(2)
    f = convbert.ConvBertFusion(IN_DIM, inner="hp", hidden_dim=HIDDEN).eval()
    a, b = _batch(), _batch()
    b.receptor = torch.randn_like(b.receptor)
    with torch.no_grad():
        assert not torch.allclose(f(a), f(b), atol=1e-6)


def test_fusion_uses_chain_structure():
    torch.manual_seed(3)
    f = convbert.ConvBertFusion(IN_DIM, inner="hp", hidden_dim=HIDDEN).eval()
    with torch.no_grad():
        assert not torch.allclose(f(_batch(chains=1)), f(_batch(chains=2)), atol=1e-6)


def test_runs_under_fp16_autocast():
    """finfo(dtype).min, not a fixed constant — a -1e9 fill would overflow fp16."""
    f = convbert.ConvBertFusion(IN_DIM, inner="hp", hidden_dim=HIDDEN).eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        out = f(_batch())
    assert torch.isfinite(out).all()


def test_is_smaller_than_the_plain_mlp_configuration_at_realistic_dims():
    """Down-projection is the point: fewer parameters than HP+MLP at 1280 dims."""
    conv = convbert.ConvBertFusion(1280, inner="hp", hidden_dim=256)
    conv_total = sum(p.numel() for p in conv.parameters()) + sum(
        p.numel() for p in heads.MLPHead(conv.out_dim).parameters()
    )
    plain = fusion.FUSIONS["hp"](1280)
    plain_total = sum(p.numel() for p in plain.parameters()) + sum(
        p.numel() for p in heads.MLPHead(plain.out_dim).parameters()
    )
    assert conv_total < plain_total
