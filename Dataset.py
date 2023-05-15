from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ShakespeareDataset(Dataset):
    def __init__(self,
                 filename=f"ShakespeareBooks/CompleteWorksOfShakespeare.txt"):
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
        sentence = self.data[item].split(" ")
        batchInputSentence = [""]
        inputSentence = ""
        for word in sentence:
            inputSentence += word + " "
            batchInputSentence.append(inputSentence.rstrip())

        # return inputSentence, outputWord
        # Todo: Tokenize the data for training generative models
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        encodedInputs = tokenizer(batchInputSentence, padding=True,
                                  truncation=True)
        encodedOutputs = tokenizer(sentence)
        return encodedInputs, encodedOutputs


if __name__ == "__main__":
    text = ShakespeareDataset()
    for i in range(1, 3):
        print(text[i])

    print(f"Total length of the dataset = {len(text)}")
