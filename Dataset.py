import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer

class ShakespeareDataset(Dataset):
    def __init__(self,
                 filename=f"ShakespeareBooks/CompleteWorksOfShakespeare.txt",
                 splitType="train"):
        super().__init__()
        self.filename = filename
        self.data = self.loadData()
        self.tokenizer = Tokenizer.from_file(path="Tokenizer/Vocab.json")
        self.trainSplits, self.valSplits = self.generateSplits()
        self.splitType = splitType

    def removeBlankLines(self, data):
        cleanedData = []
        for line in data:
            line = line.strip()
            if len(line) > 1:
                cleanedData.append(line)
        return cleanedData

    def loadData(self):
        with open(self.filename, "r") as file:
            data = file.readlines()
            data = self.removeBlankLines(data)
        return data

    def __len__(self):
        if self.splitType == "train":
            return len(self.trainSplits)
        else:
            return len(self.valSplits)

    def generateSplits(self):
        torch.manual_seed(42)
        lengthOfTheDataset = len(self.data)
        randomIndices = torch.randint(0, lengthOfTheDataset,
                                      (lengthOfTheDataset,))
        splitValue = round(0.7 * lengthOfTheDataset)
        trainIndices = randomIndices[: splitValue]
        valIndices = randomIndices[splitValue:]
        return trainIndices, valIndices

    def __getitem__(self, item):
        if self.splitType == "train":
            item = self.trainSplits[item]
        else:
            item = self.valSplits[item]
        sentence = "[CLS] " + self.data[item]
        target = self.data[item] + " [SEP]"
        tokenizedSentence = self.tokenizer.encode(sequence=sentence)
        maxSequenceLength = len(tokenizedSentence.ids)
        # return maxSequenceLength # To compute maxSequenceLength of the dataset.
        inputIds = torch.tensor([tokenizedSentence.ids] * maxSequenceLength)
        inputAttentionMask = torch.tensor([tokenizedSentence.attention_mask]
                                          * maxSequenceLength)
        sourceIds = torch.tril(inputIds)
        sourceMasks = torch.tril(inputAttentionMask)

        tokenizedTarget = self.tokenizer.encode(sequence=target)
        maxSequenceLength = len(tokenizedTarget.ids)

        targetIds = torch.tensor([tokenizedTarget.ids] * maxSequenceLength)
        targetMasks = torch.tensor([tokenizedTarget.attention_mask]
                                          * maxSequenceLength)
        targetIds = torch.tril(targetIds)
        targetMasks = torch.tril(targetMasks)

        sentenceBatch = {"sourceIds": sourceIds,
                         "sourceMasks": sourceMasks,
                         "targetIds": targetIds,
                         "targetMasks": targetMasks}
        return sentenceBatch


if __name__ == "__main__":
    text = ShakespeareDataset(splitType="val")
    for i in range(1, 3):
        batch = text[i]
        print(batch["sourceIds"])
        print(batch["sourceMasks"])