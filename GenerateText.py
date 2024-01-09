import torch
from tokenizers import Tokenizer
from Models import ShakespeareBrain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--modelName",
                    default=f"ShakespeareWith-->8Heads+CL-->512+VocabSize-->2000.pth.tar")
parser.add_argument("-nv", "--cuda", default=False, type=bool)
parser.add_argument("-t", "--tokens", default=500, type=int)
parser.add_argument("-w", "--word", default="she", type=str)
parser.add_argument("-v", "--vocabSize", default=2000, type=int)
parser.add_argument("-cl", "--contextLength", default=256, type=int)
parser.add_argument("-d", "--depth", default=8, type=int)
args = parser.parse_args()

modelName = args.modelName
cuda = args.cuda
numberOfTokens = args.tokens
firstWord = args.word
vocabSize = args.vocabSize
contextLength = args.contextLength
depth = args.depth

tokenizer = Tokenizer.from_file(path="Tokenizer/Vocab.json")
sentence = f"{firstWord} "
tokenizedSentence = tokenizer.encode(sequence=sentence)
tokenizedTarget = tokenizedSentence.ids[1:]
source = torch.tensor(tokenizedSentence.ids[:-1])
target = torch.tensor(tokenizedTarget)

if not cuda:
   checkpoint = torch.load(f"SavedModels/{modelName}", map_location="mps")
else:
    checkpoint = torch.load(f"SavedModels/{modelName}", map_location="cuda")

model = ShakespeareBrain(contextLength=contextLength,
                         classification=True,
                         numberOfHeads=8,
                         vocabSize=vocabSize,
                         generate=True,
                         depth=depth)
model.load_state_dict(checkpoint["modelStateDict"])
model.eval()

predictedTokens = torch.zeros(numberOfTokens)
predictedTokensPerLine = torch.zeros(25)
maxLength = 25
predictedWords = []
print(f"{'-'*40}\n\n")
for l in range(numberOfTokens):
    if source.size(-1) >= maxLength:
        words = tokenizer.decode(target[:-1].short().tolist())
        # predictedWords.append(words)
        print(words)
        source = source[-1:]
        target = target[-1:]

    outputs = model(source.unsqueeze(0), target.unsqueeze(0)) # Encoder-Decoder
    nextTokenProbs = outputs[-1].softmax(dim=-1)
    predictions = torch.multinomial(nextTokenProbs,
                                    num_samples=1).to("cpu")
    source = torch.cat((source, predictions))
    target = torch.cat((target, predictions))
    # print(tokenizer.decode(predictions.tolist()))

# predictedWords = tokenizer.decode(target.short().tolist())
# print("\n".join(predictedWords))

print(f"{'-'*40}")
