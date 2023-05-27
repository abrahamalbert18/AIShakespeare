from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import StripAccents
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.trainers import WordPieceTrainer

filename=f"ShakespeareBooks/CompleteWorksOfShakespeare.txt"
def trainAndSaveTokenizer(filename=f"ShakespeareBooks/CompleteWorksOfShakespeare.txt"):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.decoder = WordPieceDecoder()
    trainer = WordPieceTrainer(vocab=15000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    tokenizer.train(files=[filename], trainer=trainer)
    tokenizer.save("Tokenizer/Vocab.json")
    pass

# trainAndSaveTokenizer() # Run it only once.

def loadTokenizer(file="Tokenizer/Vocab.json"):
    return Tokenizer.from_file(path=file)

if __name__=="__main__":
    tokenizer = loadTokenizer()
    sample = tokenizer.encode("Hi, I am a new WordPiece tokenizer.")
    sampleTokens = sample.tokens
    sampleIds = sample.ids
    print(f"{sampleTokens}\n{sampleIds}")


