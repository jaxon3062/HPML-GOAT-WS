"""Optional suffix-tree lookups for SSD's async draft process (CPU only).

Two independent switches use the same per-sequence suffix trees:
  - Config.suffix_fanout: nominate one fan-out recovery token per position.
  - Config.suffix_draft:  use the tree's continuation as the speculation itself
    (SuffixDecoding as the drafter), instead of the draft model's tokens.

Keep this module free of ssd imports so it can be tested without a GPU.
"""
import torch

MAX_TREE_DEPTH = 64
# SuffixDecoding defaults (ArcticInference): draft length <= ALPHA * match length,
# stop when the estimated token probability falls below MIN_TOKEN_PROB.
ALPHA = 1.0
MIN_TOKEN_PROB = 0.1


class SuffixLookup:

    def __init__(self, K: int, min_match_len: int, global_requests: int = 0, max_seqs: int = 1024):
        # Lazy import: arctic-inference is only needed when a suffix switch is on.
        from arctic_inference.suffix_decoding import SuffixDecodingCache
        # global_requests: how many past responses the shared cross-request tree keeps.
        # 0 = per-request trees only, -1 = unlimited, N = FIFO over the last N requests.
        self.cache = SuffixDecodingCache(max_tree_depth=MAX_TREE_DEPTH, max_cached_requests=global_requests)
        self.K = K
        self.min_match_len = min_match_len
        self.max_seqs = max_seqs
        self.trunks: dict[int, list[int]] = {}  # seq_id -> accepted tokens
        self.pending: dict[int, tuple[int, list[int]]] = {}  # seq_id -> (rec, spec) not yet verified
        self.num_nominated = 0

    def start(self, seq_id: int, prompt_ids: list[int]):
        if seq_id in self.trunks:
            self._drop(seq_id)
        # shortcut: the draft process is never told a sequence finished, so evict
        # oldest-started first; upgrade to a finish message if >max_seqs run at once.
        # TODO: finish message from the target, or LRU by last use. Add behind a
        # config switch so this FIFO stays selectable.
        while len(self.trunks) >= self.max_seqs:
            self._drop(next(iter(self.trunks)))
        self.trunks[seq_id] = list(prompt_ids)
        self.cache.start_request(seq_id, prompt_ids)

    def _drop(self, seq_id: int):
        del self.trunks[seq_id]
        self.pending.pop(seq_id, None)
        self.cache.stop_request(seq_id)

    def advance(self, seq_id: int, num_accepted: int):
        """Called once per round, first. Adds last round's accepted tokens to the
        tree: the recovery token we were sent plus the num_accepted speculated
        tokens the target kept (the cache key's keep index)."""
        trunk = self.trunks.get(seq_id)
        if trunk is None or seq_id not in self.pending:
            return
        prev_rec, prev_spec = self.pending.pop(seq_id)
        accepted = [prev_rec] + prev_spec[:num_accepted]
        trunk.extend(accepted)
        self.cache.add_active_response(seq_id, accepted)

    def remember(self, seq_id: int, rec: int, spec: list[int]):
        """Called once per round, after the speculation was sent to the target."""
        if seq_id in self.trunks:
            self.pending[seq_id] = (rec, spec)

    def _context(self, seq_id: int, rec: int) -> list[int] | None:
        trunk = self.trunks.get(seq_id)
        return None if trunk is None else trunk[-MAX_TREE_DEPTH:] + [rec]

    def draft(self, seq_id: int, rec: int) -> tuple[list[int], float]:
        """SuffixDecoding speculation after recovery token rec: up to K tokens and
        their score (sum of estimated token probabilities, roughly the expected
        number accepted). ([], 0.0) if there is no match."""
        ctx = self._context(seq_id, rec)
        if ctx is None:
            return [], 0.0
        d = self.cache.speculate(seq_id, ctx, max_spec_tokens=self.K, max_spec_factor=ALPHA,
                                 min_token_prob=MIN_TOKEN_PROB)
        return list(d.token_ids), d.score

    def nominate(self, seq_id: int, rec: int, spec: list[int]) -> list[int]:
        """Returns K+1 token ids, one per position j; -1 means no nomination.
        rec/spec: this round's recovery token and the K speculated tokens sent."""
        out = [-1] * (self.K + 1)
        ctx = self._context(seq_id, rec)
        if ctx is None:  # evicted or never prefilled
            return out
        for j in range(self.K + 1):
            # The negative offset gives 0 tokens to matches shorter than
            # min_match_len, so only longer matches can nominate.
            d = self.cache.speculate(
                seq_id, ctx + spec[:j], max_spec_tokens=1,
                max_spec_factor=1.0, max_spec_offset=1.0 - self.min_match_len)
            if d.token_ids:
                out[j] = d.token_ids[0]
                self.num_nominated += 1
        return out


def pad_draft(tokens: list[int], K: int) -> list[int]:
    """Verify is fixed at K tokens. Positions past a shorter suffix draft repeat
    its last token; they only matter if every suffix token is accepted."""
    return (tokens + tokens[-1:] * K)[:K]


def boost_logits(logits: torch.Tensor, boost: torch.Tensor) -> None:
    """Sets logits[b, j, boost[b, j]] = +inf in place wherever boost[b, j] >= 0,
    so a following top-k always picks the nominated token.

    logits: [B, K+1, V], boost: [B, K+1] int64. No nonzero(), to avoid a GPU sync.
    """
    idx = boost.clamp(min=0).unsqueeze(2)
    inf = torch.full_like(logits[:, :, :1], float("inf"))
    logits.scatter_(2, idx, torch.where(boost.unsqueeze(2) >= 0, inf, logits.gather(2, idx)))
