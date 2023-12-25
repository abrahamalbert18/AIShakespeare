import torch
from tokenizers import Tokenizer
from Models import ShakespeareBrain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--modelName",
                    default=f"ShakespeareWith-->8Heads+CL-->512+VocabSize-->2000.pth.tar")
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
source = torch.tensor(tokenizedSentence.ids)
target = torch.tensor(tokenizedTarget)

if not cuda:
   modelWeights = torch.load(f"SavedModels/{modelName}", map_location="mps")
else:
    modelWeights = torch.load(f"SavedModels/{modelName}", map_location="cuda")

vocabSize = 2000
model = ShakespeareBrain(contextLength=512,
                         classification=True,
                         numberOfHeads=8,
                         vocabSize=vocabSize,
                         generate=True,
                         depth=4)
model.load_state_dict(modelWeights)
model.eval()

predictedTokens = torch.zeros(numberOfTokens)
predictedTokensPerLine = torch.zeros(25)
print(f"{'-'*40}\n\n")
for l in range(numberOfTokens//25):
    for i in range((25 - source.size(-1))//2):
        outputs = model(source.unsqueeze(0), target.unsqueeze(0))
        nextTokenProbs = outputs[-1].softmax(dim=-1)
        predictions = torch.multinomial(nextTokenProbs,
                                        num_samples=1).to("cpu")

        predictedTokensPerLine[:source.size(-1)] = source.clone()
        predictedTokensPerLine[source.size(-1) + i] = predictions.item()
        source = predictedTokensPerLine[:source.size(-1) + 1].clone()
        target = predictedTokensPerLine[1:source.size(-1) + 1].clone()
        target[-1] = 2

        # print(f"Prediction = {predictions}")
        # print(f"Source, Target: {source}, {target}")
    predictedTokens[l * 25: (l + 1) * 25] = predictedTokensPerLine
    predictedWords = tokenizer.decode(predictedTokensPerLine.short().tolist())
    print(predictedWords)
    source, target = source[-5:], target[-5:]
print(f"{'-'*40}")
