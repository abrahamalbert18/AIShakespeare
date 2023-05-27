import torch
from torch import nn
from Dataset import ShakespeareDataset

class ShakespeareBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocabSize = 30000
        self.inputSequenceLength = 32
        self.embedding = nn.Embedding(self.vocabSize, self.inputSequenceLength)
        self.transformerNetwork = nn.Transformer(nhead=8, batch_first=True,
                                                 d_model=32)

    def forward(self, encoderInputs, decoderInputs, sourceMask):
        source = self.embedding(encoderInputs.long())
        target = self.embedding(decoderInputs.long())
        outputs = self.transformerNetwork(src=source, tgt=target,)
                                          # src_mask=sourceMask)
        return outputs

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