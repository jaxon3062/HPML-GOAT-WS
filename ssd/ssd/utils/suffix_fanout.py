"""Optional suffix-tree nomination of recovery tokens for the async fan-out.

Enabled by Config.suffix_fanout. Runs in the draft process, on CPU. For each
position j of a speculation it looks up what followed the same context earlier
in the prompt or accepted output, and that token takes one fan-out slot at j.

Keep this module free of ssd imports so it can be tested without a GPU.
"""
import torch

MAX_TREE_DEPTH = 64


class SuffixFanout:

    def __init__(self, K: int, min_match_len: int, max_seqs: int = 1024):
        # Lazy import: arctic-inference is only needed when the flag is on.
        from arctic_inference.suffix_decoding import SuffixDecodingCache
        # max_cached_requests=0: per-request trees only, nothing shared across requests.
        self.cache = SuffixDecodingCache(max_tree_depth=MAX_TREE_DEPTH, max_cached_requests=0)
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
        while len(self.trunks) >= self.max_seqs:
            self._drop(next(iter(self.trunks)))
        self.trunks[seq_id] = list(prompt_ids)
        self.cache.start_request(seq_id, prompt_ids)

    def _drop(self, seq_id: int):
        del self.trunks[seq_id]
        self.pending.pop(seq_id, None)
        self.cache.stop_request(seq_id)

    def nominate(self, seq_id: int, num_accepted: int, rec: int, spec: list[int]) -> list[int]:
        """Returns K+1 token ids, one per position j; -1 means no nomination.

        num_accepted: how many of last round's spec tokens the target accepted
        (the cache key's keep index). rec/spec: this round's recovery token and
        the K speculated tokens just returned to the target.
        """
        none = [-1] * (self.K + 1)
        trunk = self.trunks.get(seq_id)
        if trunk is None:  # evicted or never prefilled
            return none
        if seq_id in self.pending:
            prev_rec, prev_spec = self.pending[seq_id]
            accepted = [prev_rec] + prev_spec[:num_accepted]
            trunk.extend(accepted)
            self.cache.add_active_response(seq_id, accepted)
        self.pending[seq_id] = (rec, spec)

        ctx = trunk[-MAX_TREE_DEPTH:] + [rec]
        out = none
        for j in range(self.K + 1):
            # The negative offset gives 0 tokens to matches shorter than
            # min_match_len, so only longer matches can nominate.
            draft = self.cache.speculate(
                seq_id, ctx + spec[:j], max_spec_tokens=1,
                max_spec_factor=1.0, max_spec_offset=1.0 - self.min_match_len)
            if draft.token_ids:
                out[j] = draft.token_ids[0]
                self.num_nominated += 1
        return out


def boost_logits(logits: torch.Tensor, boost: torch.Tensor) -> None:
    """Sets logits[b, j, boost[b, j]] = +inf in place wherever boost[b, j] >= 0,
    so a following top-k always picks the nominated token.

    logits: [B, K+1, V], boost: [B, K+1] int64. No nonzero(), to avoid a GPU sync.
    """
    idx = boost.clamp(min=0).unsqueeze(2)
    inf = torch.full_like(logits[:, :, :1], float("inf"))
    logits.scatter_(2, idx, torch.where(boost.unsqueeze(2) >= 0, inf, logits.gather(2, idx)))
