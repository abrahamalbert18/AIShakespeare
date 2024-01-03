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
import os

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

    # Padding the data with 3's
    batchSize = len(batchData)
    zeroSourceIds = 3 * torch.ones((batchSize, maxSize), dtype=torch.int16)
    zeroSourceMasks = 3 * torch.ones((batchSize, maxSize), dtype=torch.int16)
    zeroTargetIds = 3 * torch.ones((batchSize, maxSize), dtype=torch.int16)
    zeroTokensToPredict = 3 * torch.ones((batchSize, 1), dtype=torch.int16)
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

modelName = f"ShakespeareWith-->{numberOfHeads}Heads-->DepthOf{depth}+CL-->" \
            f"{contextLength}+VocabSize-->{vocabSize}.pth.tar"

trainingDataloader = DataLoader(dataset=trainingDataset, shuffle=True,
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

# optimizer
softmax = nn.Softmax()
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

# model.compile()
device = torch.device("mps") # for mac
if cuda:
    device = torch.device("cuda:1") # for NVIDIA GPUs

# best metrics and parameters
bestEpoch = 0
bestEpochLoss = 10

# Load from checkpoint
if os.path.exists(f"SavedModels/{modelName}"):
    checkpoint = torch.load(f"SavedModels/{modelName}",
                            map_location=device)
    model.load_state_dict(checkpoint["modelStateDict"])
    bestEpochLoss = checkpoint["bestEpochLoss"]
    learningRate = checkpoint["learningRate"]
    # optimizer.load_state_dict(checkpoint["optimizerStateDict"])
    del checkpoint

model.to(device)
"""
# Code for distributed computing
if distributed.is_available():
    # do something
"""


# optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
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
                    optimizer.zero_grad()

            epochLoss += loss.item()
        # scheduler.step()
        """
        Epoch metrics
        """
        averageEpochLoss = epochLoss / (e + 1)
        if epoch % 1 == 0:
            print(f"{phase} loss = {averageEpochLoss:.4f}")
        writer.add_scalar(f"{phase.capitalize()} Loss/Epoch", averageEpochLoss,
                          epoch + 1)
        if (averageEpochLoss < bestEpochLoss) and phase == "val":
            bestEpochLoss = averageEpochLoss
            bestEpoch = epoch
            torch.save({"epoch": epoch+1,
                        "modelStateDict": model.state_dict(),
                        # "optimizerStateDict":optimizer.state_dict(),
                        "bestEpochLoss":round(bestEpochLoss, 4),
                        "learningRate":learningRate},
                       f"SavedModels/{modelName}")
        writer.close()
print(f"Best loss: {round(bestEpochLoss, 4)} @ epoch #{bestEpoch + 1}")
print(f"Best model saved.")