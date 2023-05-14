def loadData(numberOfLines):
    with open(f"ShakespeareBooks/CompleteWorksOfShakespeare.txt", "r") as file:
        data = file.readlines()
        data = removeBlankLines(data)
        print(data[:numberOfLines])

def removeBlankLines(data):
    cleanedData = []
    for line in data:
        line = line.strip()
        if len(line) > 1:
            cleanedData.append(line)
    return cleanedData

if __name__=="__main__":
    loadData(numberOfLines=100)