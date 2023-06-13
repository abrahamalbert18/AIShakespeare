import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from Dataset import ShakespeareDataset
from Models import ShakespeareBrain
from tqdm import tqdm
import argparse

torch.manual_seed(42)
# For generating outputs
tokenizer = Tokenizer.from_file(path="Tokenizer/Vocab.json")

trainingDataset = ShakespeareDataset(splitType="train")
validationDataset = ShakespeareDataset(splitType="val")

lengthOfDatasets = {"train": len(trainingDataset),
                    "val": len(validationDataset)}

# define custom collator for batches
def customCollator(batchData):
    maxSize = max([batchData[i]["sourceIds"].shape[0]
                   for i in range(len(batchData))])

    # Padding the data
    zeroSourceIds = torch.zeros((maxSize, maxSize), dtype=torch.int16)
    zeroSourceMasks = torch.zeros((maxSize, maxSize), dtype=torch.int16)
    zeroTargetIds = torch.zeros((maxSize, maxSize), dtype=torch.int16)
    for item in batchData:
        zeroSourceIds[:item["sourceIds"].size(0), :item["sourceIds"].size(-1)] = item["sourceIds"]
        zeroSourceMasks[:item["sourceMasks"].size(0), :item["sourceMasks"].size(-1)] = item["sourceMasks"]
        zeroTargetIds[:item["targetIds"].size(0), :item["targetIds"].size(-1)] = item["targetIds"]

    return zeroSourceIds, zeroTargetIds, zeroSourceMasks.float()

parser = argparse.ArgumentParser()
parser.add_argument("-batchSize", "-bs", default=40, type=int)
parser.add_argument("-learningRate", "-lr", default=1e-5, type=float)
parser.add_argument("-contextLength", "-cl", default=32, type=int)
args = parser.parse_args()

batchSize = args.batchSize
learningRate = args.learningRate
contextLength = args.contextLength

trainingDataloader = DataLoader(dataset=trainingDataset, shuffle=True,
                                batch_size=batchSize,
                                collate_fn=customCollator)

validationDataloader = DataLoader(dataset=validationDataset, shuffle=True,
                                  batch_size=batchSize,
                                  collate_fn=customCollator)

dataloaders = {"train": trainingDataloader,
               "val": validationDataloader}

device = torch.device("mps")
model = ShakespeareBrain(contextLength)
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax()
optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

# learning rate scheduler
numberOfEpochs = 20

# best metrics and parameters
bestEpoch = 0
# bestModelWeights = copy.deepcopy(model.state_dict)

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
        for e, batch in enumerate(dataloaders[phase]):
            sourceIds, targetIds, sourceMasks = batch[0], batch[1], batch[2]
            sourceIds = sourceIds.to(device)
            sourceMasks = sourceMasks.to(device)
            targetIds = targetIds.to(device)

            with torch.set_grad_enabled(phase == "train"):
                outputs, loss = model(sourceIds, targetIds, sourceMasks)
                predictions = outputs.softmax(dim=1).max(-1)[1].to("cpu")
                if e % 20 == 0:
                    predictedTargets = batch[1].clone() # gets updated
                    for i in range(predictedTargets.shape[0]):
                        # print(predictedTargets[i, i])
                        predictedTargets[i, i] = predictions[i]
                        # print(predictedTargets[i, i])
                    predictedText = tokenizer.decode_batch(
                                        predictedTargets.tolist())
                    predictedText = '\n'.join(predictedText)
                    originalText = tokenizer.decode_batch(
                                        batch[1].tolist())
                    originalText = '\n'.join(originalText)
                    print(f"Actual Targets:\n{originalText}")
                    print(f"Predictions:\n{predictedText}")

                if phase == "train":
                    # backpropgate the loss
                    loss.backward()
                    # update the weights
                    optimizer.step()
                    optimizer.zero_grad()
            #TODO write code for generating text predictions
            epochLoss += loss

        """
        Epoch metrics
        """
        averageEpochLoss = epochLoss / (e + 1)
        print(f"{phase} loss = {averageEpochLoss:.4f}")
