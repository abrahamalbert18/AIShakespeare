import torch
from torch import nn
from Dataset import ShakespeareDataset

class ShakespeareBrain(nn.Module):
    def __init__(self, numberOfHeads=4, contextLength=32, classification=True,
                 vocabSize=2000, generate=False, depth=6):
        super().__init__()
        self.vocabSize = vocabSize
        self.contextLength = contextLength
        self.numberOfHeads = numberOfHeads
        self.classifcation = classification
        self.generate = generate
        self.depth = depth
        # self.oneHotEncoding = torch.zeros(self.vocabSize, self.vocabSize)
        self.wordEmbedding = nn.Embedding(self.vocabSize, self.contextLength)
        self.positionEmbedding = nn.Embedding(self.vocabSize,
                                              self.contextLength)
        self.transformerNetwork = nn.Transformer(nhead=self.numberOfHeads,
                                                 batch_first=True,
                                                 d_model=self.contextLength,
                                                 num_encoder_layers=self.depth,
                                                 num_decoder_layers=self.depth)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=self.contextLength,
                                                        nhead=self.numberOfHeads,
                                                       batch_first=True,
                                                       dropout=0.2)
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=self.contextLength,
                                                        nhead=self.numberOfHeads,
                                                       batch_first=True,
                                                       dropout=0.2)
        self.encoderNetwork = nn.TransformerEncoder(encoder_layer=self.encoderLayer,
                                                    num_layers=self.depth)
        self.decoderNetwork = nn.TransformerDecoder(decoder_layer=self.decoderLayer,
                                                    num_layers=self.depth)
        if self.classifcation:
            self.criterion = nn.CrossEntropyLoss(ignore_index=3,
                                                 reduction="mean")
            #classification
        else:
            self.criterion = nn.MSELoss()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layerNorm = nn.LayerNorm(self.contextLength)
        self.predictionLayer = nn.Linear(self.contextLength,
                                self.vocabSize)

    def forward(self, encoderInputs, decoderInputs, sourceMask=None,
                tokensToPredict=None):
        B, T = encoderInputs.size()
        targetTokens = tokensToPredict
        position = torch.arange(0, T, device=encoderInputs.device,
                                dtype=torch.long)
        source = self.layerNorm(self.wordEmbedding(encoderInputs.long())) + \
                 self.positionEmbedding(position)
        target = self.layerNorm(self.wordEmbedding(decoderInputs.long())) + \
                 self.positionEmbedding(position)
        # outputs = self.transformerNetwork(src=source, tgt=target)
        outputs = self.encoderNetwork(src=source)
        # outputs = self.decoderNetwork(tgt=source, memory=target)
        # outputs = self.layerNorm(outputs)
        outputs = self.predictionLayer(outputs) # B, T, VocabSize
        outputs = outputs.view(-1, outputs.size(-1)) # B * T, VocabSize
        if self.generate:
            return outputs

        # loss = self.criterion(outputs, targetTokens.long().view(-1))
        loss = self.criterion(outputs, decoderInputs.long().view(-1))
        return outputs, loss

if __name__=="__main__":
    model = ShakespeareBrain()

    def simpleTest(index=1):
        text = ShakespeareDataset(splitType="val",
                                  filename=f"ShakespeareBooks/ShakespeareTexts.txt")
        pass
    outputs = simpleTest()
    # for i in range(800,1800):
    #     simpleTest(i)
