from Dataset import ShakespeareDataset
from tqdm import tqdm

data = ShakespeareDataset()
maximumSequenceLength = 0

for i in tqdm(range(len(data)), desc="Progress"):
    currentSequenceLength = data[i]
    if currentSequenceLength > maximumSequenceLength:
        maximumSequenceLength = currentSequenceLength
    if i % 1000 == 0:
        print(f"Maximum sequence length after {i} iterations : "
              f"{maximumSequenceLength}")

print(f"Maximum sequence length of the dataset = {maximumSequenceLength}")