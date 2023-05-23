import torch
from torch import nn

class ShakespeareBrain(nn.Module):
    def __init__(self, maxInputSequenceLength, numOfOutputWords):
        super().__init__()
        self.numOfOutputs = numOfOutputWords
        self.inputSequenceLength = maxInputSequenceLength
        self.transformerNetwork = nn.Transformer(nhead=8, batch_first=True)

    def forward(self, encoderInputs, decoderInputs):
        outputs = self.transformerNetwork(encoderInputs, decoderInputs)
        return outputs

if __name__=="__main__":
    model = ShakespeareBrain(20,20)
