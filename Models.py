import torch
from torch import nn
from Dataset import ShakespeareDataset

class ShakespeareBrain(nn.Module):
    def __init__(self, numberOfHeads=4, contextLength=32, classification=True,
                 vocabSize=2000):
        super().__init__()
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.numberOfHeads = numberOfHeads
        self.classifcation = classification
        # self.oneHotEncoding = torch.zeros(self.vocabSize, self.vocabSize)
        self.wordEmbedding = nn.Embedding(self.vocabSize, self.contextLength)
        self.positionEmbedding = nn.Embedding(self.vocabSize,
                                              self.contextLength)
        self.transformerNetwork = nn.Transformer(nhead=self.numberOfHeads,
                                                 batch_first=True,
                                                 d_model=self.contextLength)
        if self.classifcation:
            self.criterion = nn.CrossEntropyLoss() #classification
        else:
            self.criterion = nn.MSELoss(reduction="sum")
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layerNorm = nn.LayerNorm(self.contextLength)

    def forward(self, encoderInputs, decoderInputs, sourceMask=None):
        B, T = encoderInputs.size()
        targetTokens = decoderInputs.diag()
        position = torch.arange(0, T, device=encoderInputs.device,
                                dtype=torch.long)
        source = self.layerNorm(self.wordEmbedding(encoderInputs.long())) + \
                 self.positionEmbedding(position)
        target = self.layerNorm(self.wordEmbedding(decoderInputs.long())) + \
                 self.positionEmbedding(position)
        outputs = self.transformerNetwork(src=source, tgt=target)

        if not self.classifcation:
            # Regression
            predictionLayer = nn.Linear(T * self.contextLength, 1).to(
                                        encoderInputs.device)

            outputs = predictionLayer(outputs.view(B, -1)).squeeze()
            # outputs = self.relu(outputs)
            targetTokens = torch.div(targetTokens, self.vocabSize)
        else:
        # Classification
            predictionLayer = nn.Linear(T * self.contextLength,
                                self.vocabSize).to(encoderInputs.device)
            outputs = predictionLayer(outputs.view(B, -1))

        loss = self.criterion(outputs, targetTokens)
        return outputs, loss

if __name__=="__main__":
    model = ShakespeareBrain()

    def simpleTest(index=1):
        text = ShakespeareDataset()
        inputs = text[index]
        source = inputs["sourceIds"]
        target = inputs["targetIds"]
        sourceMasks = inputs["sourceMasks"].bool()
        outputs = model(source, target, sourceMasks)
        print(outputs.shape)
        return outputs
    # outputs = simpleTest()
    # for i in range(800,1800):
    #     simpleTest(i)