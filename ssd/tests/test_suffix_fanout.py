"""CPU-only tests for the suffix fan-out logic. Run from the repo root:

    python -m pytest tests/test_suffix_fanout.py

Loads suffix_fanout.py by path so `import ssd` (which needs CUDA libs) is not triggered.
"""
import importlib.util
from pathlib import Path

import torch

_path = Path(__file__).resolve().parents[1] / "ssd" / "utils" / "suffix_fanout.py"
_spec = importlib.util.spec_from_file_location("suffix_fanout", _path)
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)


def round_(f, seq_id, num_accepted, rec, spec):
    """One draft-process round as draft_runner does it: advance, remember, nominate."""
    f.advance(seq_id, num_accepted)
    f.remember(seq_id, rec, spec)
    return f.nominate(seq_id, rec, spec)


def test_nominates_continuation_from_prompt():
    f = sf.SuffixLookup(K=3, min_match_len=2)
    f.start(0, [10, 11, 12, 13, 14, 15, 99])
    # context ends "... 99, 10" + spec[:j]; only j>=1 gives a 2-token match ("10 11").
    assert round_(f, 0, -2, rec=10, spec=[11, 12, 13]) == [-1, 12, 13, 14]
    assert f.num_nominated == 3


def test_min_match_len_gates_short_matches():
    f = sf.SuffixLookup(K=1, min_match_len=3)
    f.start(0, [10, 11, 12, 13, 99])
    assert round_(f, 0, -2, rec=10, spec=[11]) == [-1, -1]  # longest match is 2
    f = sf.SuffixLookup(K=1, min_match_len=1)
    f.start(0, [10, 11, 12, 13, 99])
    assert round_(f, 0, -2, rec=10, spec=[11]) == [11, 12]


def test_accepted_tokens_enter_the_tree_and_rejected_do_not():
    f = sf.SuffixLookup(K=2, min_match_len=2)
    f.start(0, [1, 2, 3])
    round_(f, 0, -2, rec=50, spec=[51, 52])
    # Target accepted 1 of the 2 spec tokens, then produced recovery token 60.
    round_(f, 0, 1, rec=60, spec=[61, 62])
    assert f.trunks[0] == [1, 2, 3, 50, 51]
    round_(f, 0, 0, rec=50, spec=[51, 7])
    assert f.trunks[0] == [1, 2, 3, 50, 51, 60]
    # "50 51" was accepted earlier and was followed by 60; rejected 52 must not appear.
    assert round_(f, 0, 2, rec=50, spec=[51, 8])[1] == 60


def test_advance_folds_each_round_once():
    f = sf.SuffixLookup(K=2, min_match_len=1)
    f.start(0, [1])
    f.remember(0, 5, [6, 7])
    f.advance(0, 2)
    f.advance(0, 2)  # nothing pending any more: no double insert
    assert f.trunks[0] == [1, 5, 6, 7]


def test_unknown_and_evicted_sequences_get_nothing():
    f = sf.SuffixLookup(K=2, min_match_len=1, max_seqs=2)
    assert round_(f, 7, -2, 1, [2, 3]) == [-1, -1, -1]
    assert f.draft(7, 1) == ([], 0.0)
    f.start(0, [1, 2, 3])
    f.start(1, [1, 2, 3])
    f.start(2, [1, 2, 3])  # evicts seq 0
    assert set(f.trunks) == {1, 2}
    assert round_(f, 0, -2, 1, [2, 3]) == [-1, -1, -1]
    f.start(1, [4, 5, 6])  # restarting an active id must not raise
    assert f.trunks[1] == [4, 5, 6]


def test_global_tree_is_off_by_default_and_shares_responses_when_on():
    def other_request_sees(global_requests):
        f = sf.SuffixLookup(K=2, min_match_len=2, global_requests=global_requests)
        f.start(0, [1, 2, 3])
        round_(f, 0, -2, rec=50, spec=[51, 52])
        round_(f, 0, 2, rec=53, spec=[7, 8])  # 50 51 52 accepted by seq 0
        f.start(1, [4, 5, 6])
        return round_(f, 1, -2, rec=50, spec=[51, 9])[1]  # context "... 50 51"

    assert other_request_sees(0) == -1
    assert other_request_sees(-1) == 52


def test_draft_returns_repeated_continuation_with_score():
    f = sf.SuffixLookup(K=4, min_match_len=3)
    f.start(0, [9, 1, 2, 3, 4, 5, 6, 8, 1, 2])
    # context "... 1 2" + rec 3 matches "1 2 3" (3 tokens) -> continuation 4 5 6,
    # capped at ALPHA * match length = 3 tokens.
    toks, score = f.draft(0, rec=3)
    assert toks == [4, 5, 6]
    assert score == 3.0  # every step seen once, probability 1


def test_draft_is_empty_without_a_match():
    f = sf.SuffixLookup(K=4, min_match_len=3)
    f.start(0, [1, 2, 3, 4])
    assert f.draft(0, rec=77) == ([], 0.0)


def test_pad_draft_fills_to_K():
    assert sf.pad_draft([4, 5], 4) == [4, 5, 5, 5]
    assert sf.pad_draft([4, 5, 6, 7, 8], 4) == [4, 5, 6, 7]


def test_boost_logits_forces_topk_and_leaves_other_rows_alone():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 20)
    before = logits.clone()
    boost = torch.tensor([[5, -1, 7], [-1, -1, 0]])
    sf.boost_logits(logits, boost)
    top1 = logits.argmax(-1)
    assert top1[0, 0] == 5 and top1[0, 2] == 7 and top1[1, 2] == 0
    untouched = boost < 0
    assert torch.equal(logits[untouched], before[untouched])
    # Exactly one entry changed per boosted row.
    assert ((logits != before).sum(-1) == (~untouched).long()).all()


def test_boost_then_exclusion_matches_fork_semantics():
    # Mirrors get_forked_recovery_tokens_from_logits: boost, then -inf on the
    # draft's own token. A nomination equal to the draft's pick must not survive.
    logits = torch.zeros(1, 2, 10)
    logits[0, 0, 3] = 1.0
    sf.boost_logits(logits, torch.tensor([[4, 4]]))
    logits[:, :-1, :] = logits[:, :-1, :].scatter(2, torch.tensor([[[4]]]), float("-inf"))
    assert logits[0, 0].argmax() == 3
    assert logits[0, 1].argmax() == 4
