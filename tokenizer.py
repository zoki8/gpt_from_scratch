
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


if __name__ == "__main__":
    text = open("data/input.txt", encoding="utf-8").read()
    tok = CharTokenizer(text)
    print(tok.vocab_size)              
    ids = tok.encode("Hello there!")
    print(ids)
    print(tok.decode(ids))            
    assert tok.decode(tok.encode(text[:1000])) == text[:1000]