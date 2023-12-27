import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from Dataset import ShakespeareDataset
from Models import ShakespeareBrain
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as distributed

torch.manual_seed(42)
# For generating outputs
tokenizer = Tokenizer.from_file(path="Tokenizer/Vocab.json")

trainingDataset = ShakespeareDataset(splitType="train", filename=f"ShakespeareBooks/ShakespeareTexts.txt")
validationDataset = ShakespeareDataset(splitType="val", filename=f"ShakespeareBooks/ShakespeareTexts.txt")

lengthOfDatasets = {"train": len(trainingDataset),
                    "val": len(validationDataset)}

# define custom collator for batches
def customCollator(batchData):
    maxSize = max([batchData[i]["sourceIds"].shape[0]
                   for i in range(len(batchData))])

    # Padding the data
    batchSize = len(batchData)
    zeroSourceIds = torch.zeros((batchSize, maxSize), dtype=torch.int16)
    zeroSourceMasks = torch.zeros((batchSize, maxSize), dtype=torch.int16)
    zeroTargetIds = torch.zeros((batchSize, maxSize), dtype=torch.int16)
    zeroTokensToPredict = torch.zeros((batchSize, 1), dtype=torch.int16)
    for i, item in enumerate(batchData):
        zeroSourceIds[i,:item["sourceIds"].size(-1)] = item["sourceIds"]
        zeroSourceMasks[i, :item["sourceMasks"].size(-1)] = item["sourceMasks"]
        zeroTargetIds[i, :item["targetIds"].size(-1)] = item["targetIds"]
        zeroTokensToPredict[i, 0] = item["tokensToPredict"]
    return zeroSourceIds, zeroTargetIds, zeroSourceMasks.float(), zeroTokensToPredict

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchSize",  default=200, type=int)
parser.add_argument("-lr", "--learningRate", default=1e-3, type=float)
parser.add_argument("-cl", "--contextLength", default=256, type=int)
# parser.add_argument("-h", "--numberOfHeads", default=4, type=int)
parser.add_argument("-e", "--epochs", default=20, type=int)
parser.add_argument("-c", "--classification", default=True, type=bool)
parser.add_argument("-nv", "--cuda", default=False, type=bool)
parser.add_argument("-v", "--vocabSize", default=2000, type=int)
parser.add_argument("-d", "--depth", default=8, type=int)
args = parser.parse_args()

batchSize = args.batchSize
learningRate = args.learningRate
contextLength = args.contextLength
numberOfEpochs = args.epochs
isClassification = args.classification
cuda = args.cuda
# numberOfHeads = args.numberOfHeads
numberOfHeads = 8
depth = args.depth
vocabSize = args.vocabSize

modelName = f"ShakespeareWith-->{numberOfHeads}Heads+CL-->" \
            f"{contextLength}+VocabSize-->{vocabSize}.pth.tar"

trainingDataloader = DataLoader(dataset=trainingDataset, shuffle=False,
                                batch_size=batchSize,
                                collate_fn=customCollator)

validationDataloader = DataLoader(dataset=validationDataset, shuffle=True,
                                  batch_size=batchSize,
                                  collate_fn=customCollator)

dataloaders = {"train": trainingDataloader,
               "val": validationDataloader}

model = ShakespeareBrain(contextLength=contextLength,
                         classification=isClassification,
                         numberOfHeads=numberOfHeads,
                         vocabSize=vocabSize,
                         generate=False,
                         depth=depth)
# model.compile()
device = torch.device("mps") # for mac
if cuda:
    device = torch.device("cuda:1") # for NVIDIA GPUs

model.to(device)

"""
# Code for distributed computing
if distributed.is_available():
    # do something
"""

# loss and optimizer
# criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# learning rate scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# best metrics and parameters
bestEpoch = 0
bestEpochLoss = 13.0
writer = SummaryWriter(f"runs/{modelName}")

# training and evaluation loop
for epoch in tqdm(range(numberOfEpochs), desc="Epoch progress:", leave=False):
    print("-" * 40)
    print(f"Epoch {epoch + 1}:")
    # Setting phase
    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        epochLoss = 0
        for e, batch in tqdm(enumerate(dataloaders[phase]),
                             desc="Iteration progress",
                             leave=False):
            sourceIds, targetIds, sourceMasks = batch[0], batch[1], batch[2]
            tokensToPredict = batch[-1]
            sourceIds = sourceIds.to(device)
            sourceMasks = sourceMasks.to(device)
            targetIds = targetIds.to(device)
            tokensToPredict = tokensToPredict.to(device)

            with torch.set_grad_enabled(phase == "train"):
                outputs, loss = model(sourceIds, targetIds, sourceMasks,
                                      tokensToPredict)

                if phase == "train":
                    # backpropgate the loss
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()

            epochLoss += loss

        """
        Epoch metrics
        """
        averageEpochLoss = epochLoss / (e + 1)
        if epoch % 20 == 0:
            print(f"{phase} loss = {averageEpochLoss:.4f}")
        writer.add_scalar(f"{phase.capitalize()} Loss/Epoch", averageEpochLoss,
                          epoch + 1)
        if (averageEpochLoss < bestEpochLoss) and phase == "val":
            bestEpochLoss = averageEpochLoss
            bestEpoch = epoch
            torch.save(model.state_dict(), f"SavedModels/{modelName}")
            torch.save(optimizer.state_dict(), f"SavedModels/OptimizerFor"
                                               f"{modelName}")
        writer.close()
print(f"Best loss: {torch.round(bestEpochLoss, 4)} @ epoch #{bestEpoch + 1}")
print(f"Best model saved.")