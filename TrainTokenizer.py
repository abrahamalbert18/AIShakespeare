from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import StripAccents
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.decoders import ByteLevel as BytePieceDecoder
from tokenizers.trainers import WordPieceTrainer
from tokenizers.trainers import BpeTrainer
from tokenizers import processors
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-vs", "--vocabSize", default=5000, type=int)
parser.add_argument("-f", "--filename",
                    default = f"ShakespeareBooks/ShakespeareTexts.txt",
                    type = str)
args = parser.parse_args()
vocabSize = args.vocabSize
filename = args.filename

def trainAndSaveTokenizer(
                    filename=f"ShakespeareBooks/ShakespeareTexts.txt",
                    vocabSize=vocabSize):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = WordPieceDecoder()
    trainer = WordPieceTrainer(vocab_size=vocabSize,
                               special_tokens=["[UNK]","[CLS]",
                                               "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train(files=[filename], trainer=trainer)
    clsTokenId = tokenizer.token_to_id("[CLS]")
    sepTokenId = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS]:0 $A:0 [SEP]:0",
            pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[("[CLS]", clsTokenId), ("[SEP]", sepTokenId)],
    )
    tokenizer.save("Tokenizer/Vocab.json")
    pass

trainAndSaveTokenizer(filename, vocabSize) # Run it only once.

def loadTokenizer(file="Tokenizer/Vocab.json"):
    return Tokenizer.from_file(path=file)

if __name__=="__main__":
    tokenizer = loadTokenizer()
    sample = tokenizer.encode("Hi, I am a new WordPiece tokenizer.")
    sampleTokens = sample.tokens
    sampleIds = sample.ids
    print(f"{sampleTokens}\n{sampleIds}")


