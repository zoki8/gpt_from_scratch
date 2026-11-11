from hmac import new
import re
from collections import Counter

PRETOKENIZE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {char : i for i,char in enumerate(self.chars)}
        self.itos = {i : char for i,char in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        encodeList=[self.stoi[ns] for ns in s]
        return encodeList

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids)

    def to_dict(self) -> dict:
        return {'type': 'char', 'chars': self.chars}


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i<len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

class BPETokenizer:
    def __init__(self, chars: list[str], merges: list[tuple[int, int]]):
        self.chars = list(chars)
        self.merges = [tuple(m) for m in merges]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab = {i: c for i, c in enumerate(self.chars)}
        self.ranks = {}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[len(self.chars) + rank] = self.vocab[a] + self.vocab[b]
            self.ranks[(a, b)] = rank
        self.cache = {}

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        words = Counter(PRETOKENIZE.findall(text))
        seqs = {w: [stoi[c] for c in w] for w in words}
        merges = []
        for new_id in range(len(chars), vocab_size):
            pairs = Counter()
            for w, freq in words.items():
                s = seqs[w]
                for pair in zip(s,s[1:]):
                    pairs[pair] += freq
            
            if not pairs:
                break

            best = max(pairs, key = pairs.get)

            merges.append(best)

            for w in words:
                seqs[w] = merge(seqs[w], best, new_id)

        return cls(chars, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = [self.stoi[c] for c in chunk]
        while len(ids) >= 2:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float('inf')))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, len(self.chars) + self.ranks[pair])
        return ids

    def encode(self, s: str) -> list[int]:
        out = []
        for chunk in PRETOKENIZE.findall(s):
            if chunk not in self.cache:
                self.cache[chunk] = self._encode_chunk(chunk)
            out.extend(self.cache[chunk])
        return out

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.vocab[i] for i in ids)

    def to_dict(self) -> dict:
        return {'type': 'bpe', 'chars': self.chars, 'merges': [list(m) for m in self.merges]}

def load_tokenizer(d: dict):
    if d['type'] == 'char':
        return CharTokenizer(''.join(d['chars']))
    return BPETokenizer(d['chars'], d['merges'])


if __name__ == "__main__":
    text = open("data/input.txt", encoding="utf-8").read()

    # provera za TODO 1
    assert merge([5, 6, 7, 5, 6], (5, 6), 99) == [99, 7, 99]
    assert merge([5, 5, 5], (5, 5), 9) == [9, 5]
    print("merge: OK")

    # provera za TODO 2-6
    tok = BPETokenizer.train(text, 512)
    print("vocab:", tok.vocab_size)
    ids = tok.encode("ROMEO: Wherefore art thou Romeo?")
    print([tok.decode([i]) for i in ids])
    assert tok.decode(tok.encode(text[:20000])) == text[:20000]
    print("decode(encode(tekst)) == tekst: OK")
