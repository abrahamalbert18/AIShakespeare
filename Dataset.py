import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer

class ShakespeareDataset(Dataset):
    def __init__(self,
                 filename=f"ShakespeareBooks/CompleteWorksOfShakespeare.txt"):
        super().__init__()
        self.filename = filename
        self.data = self.loadData()
        self.tokenizer = Tokenizer.from_file(path="Tokenizer/Vocab.json")

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
        return len(self.data)

    def __getitem__(self, item):
        sentence = "[CLS] " + self.data[item] + " [SEP]"
        tokenizedSentence = self.tokenizer.encode(sequence=sentence)
        maxSequenceLength = len(tokenizedSentence.ids)
        # return maxSequenceLength # To compute maxSequenceLength of the dataset.
        inputIds = torch.tensor([tokenizedSentence.ids] * maxSequenceLength)
        inputAttentionMask = torch.tensor([tokenizedSentence.attention_mask]
                                          * maxSequenceLength)
        sourceIds = torch.tril(inputIds)
        sourceMasks = torch.tril(inputAttentionMask)

        targetIds = torch.diag(sourceIds).unsqueeze(dim=1)
        targetMasks = torch.diag(sourceMasks).unsqueeze(dim=1)
        sentenceBatch = {"sourceIds": sourceIds,
                         "sourceMasks": sourceMasks,
                         "targetIds": targetIds,
                         "targetMasks": targetMasks}
        return sentenceBatch


if __name__ == "__main__":
    text = ShakespeareDataset()
    for i in range(1, 3):
        batch = text[i]
        print(batch["ids"])
        print(batch["masks"])