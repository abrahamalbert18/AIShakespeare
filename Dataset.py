import torch
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, filename = f"ShakespeareBooks/CompleteWorksOfShakespeare.txt"):
        super().__init__()
        self.filename = filename
        self.data = self.loadData()

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
        return self.data[item]

if __name__=="__main__":
    text = ShakespeareDataset()
    for i in range(10):
        print(text[i])

    print(f"Total length of the dataset = {len(text)}")