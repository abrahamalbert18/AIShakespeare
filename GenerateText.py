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

modelWeights = torch.load(f"SavedModels/{modelName}", map_location="mps")
if cuda:
    model = torch.load(f"SavedModels/{modelName}", map_location="cuda")

vocabSize = 10000
model = ShakespeareBrain(contextLength=512,
                         classification=False,
                         numberOfHeads=8,
                         vocabSize=vocabSize,
                         generate=True)
model.load_state_dict(modelWeights)
model.eval()

predictedTokens = torch.zeros(numberOfTokens)

print(f"{'-'*40}\n\n")
for _ in range(numberOfTokens//25):
    for i in range(25):
        outputs = model(source.unsqueeze(0), target.unsqueeze(0))
        predictions = outputs.mul(vocabSize).to("cpu").round()
        if predictions.data == vocabSize:
           continue
        predictedTokens[i], predictedTokens[i+1] = predictions.data, 2
        source = predictedTokens[:i+1]
        target = predictedTokens[1:i+2]
        # print(predictions)
        # print(source, target)

    predictedWords = tokenizer.decode(source.short().tolist())
    print(predictedWords)
    source, target = source[i-5:], target[i-5:]
print(f"{'-'*40}")