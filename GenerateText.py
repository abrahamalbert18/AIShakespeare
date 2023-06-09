import torch
from tokenizers import Tokenizer
from Models import ShakespeareBrain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--modelName",
                    default=f"ShakespeareWith-->8Heads+CL-->512+VocabSize-->10000.pth.tar")
parser.add_argument("-nv", "--cuda", default=False, type=bool)
parser.add_argument("-t", "--tokens", default=500, type=int)
parser.add_argument("-w", "--word", default="love", type=str)

args = parser.parse_args()

modelName = args.modelName
cuda = args.cuda
numberOfTokens = args.tokens
firstWord = args.word

tokenizer = Tokenizer.from_file(path="Tokenizer/Vocab.json")
sentence = f"{firstWord} "
tokenizedSentence = tokenizer.encode(sequence=sentence)
tokenizedTarget = tokenizedSentence.ids[1:] + [2]
if len(tokenizedSentence.ids) > 1:
    tokenizedTarget = tokenizedSentence.ids[1:]
source = torch.tensor(tokenizedSentence.ids)
target = torch.tensor(tokenizedTarget)

if not cuda:
   modelWeights = torch.load(f"SavedModels/{modelName}", map_location="mps")
else:
    modelWeights = torch.load(f"SavedModels/{modelName}", map_location="cuda")

vocabSize = 10000
model = ShakespeareBrain(contextLength=512,
                         classification=False,
                         numberOfHeads=8,
                         vocabSize=vocabSize,
                         generate=True,
                         depth=3)
model.load_state_dict(modelWeights)
model.eval()

predictedTokens = torch.zeros(numberOfTokens)
predictedTokensPerLine = torch.zeros(25)
print(f"{'-'*40}\n\n")
for l in range(numberOfTokens//25):
    for i in range(25):
        outputs = model(source.unsqueeze(0), target.unsqueeze(0))
        predictions = outputs.mul(vocabSize).to("cpu").round()
        if predictions.data == vocabSize:
           continue
        predictedTokensPerLine[i] = predictions.data
        source = predictedTokensPerLine[:i + 1]
        target = predictedTokensPerLine[1:i + 2]
        target[-1] = 2
        # print(predictions)
        # print(source, target
    predictedTokens[l * 25: (l + 1) * 25] = predictedTokensPerLine
    predictedWords = tokenizer.decode(predictedTokensPerLine.short().tolist())
    print(predictedWords)
    t = torch.zeros(predictedTokensPerLine.shape)
    t[:target.size(-1)] = target
    t[-1] = 2
    source, target = source[-5:-1], t[-5:-1]
print(f"{'-'*40}")
