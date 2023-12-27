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
            # data = self.removeBlankLines(data)
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
        splitValue = round(0.85 * lengthOfTheDataset)
        trainIndices = randomIndices[: splitValue]
        valIndices = randomIndices[splitValue:]
        return trainIndices, valIndices

    def __getitem__(self, item):
        if self.splitType == "train":
            item = self.trainSplits[item]
        else:
            item = self.valSplits[item]
        sentence = self.data[item]
        if (item + 1) >= len(self.data):
            nextSentence = "\n"
        else:
            nextSentence = self.data[item + 1]
        tokenizedSentence = self.tokenizer.encode(sequence=sentence)
        tokenizedNextSentence = self.tokenizer.encode(nextSentence)
        tokenToPredict = tokenizedNextSentence.ids[1]
        decoderInputIds = tokenizedSentence.ids[1:]
        decoderInputIds.append(tokenToPredict)
        sentenceBatch = {"sourceIds": torch.tensor(tokenizedSentence.ids),
                         "sourceMasks": torch.tensor(
                                 tokenizedSentence.attention_mask),
                         "targetIds": torch.tensor(decoderInputIds),
                         "tokensToPredict":
                             torch.tensor(tokenToPredict).unsqueeze(0)}
        return sentenceBatch


if __name__ == "__main__":
    text = ShakespeareDataset(splitType="val", filename=f"ShakespeareBooks/ShakespeareTexts.txt")
    for i in range(len(text)-5, len(text)):
        print(f"Number: {i+1}")
        batch = text[i]
        print(batch["sourceIds"])
        print(batch["targetIds"])
        print(batch["sourceMasks"])
        print(i, batch["tokensToPredict"])
        print(text.tokenizer.decode(batch["sourceIds"].tolist()))
