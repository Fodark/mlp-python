from mlp import Mlp
import random
import pandas as pd

def xor():
    training = [
        {
            "input": [0,0],
            "output": [0]
        },
        {
            "input": [1,0],
            "output": [1]
        },
        {
            "input": [0,1],
            "output": [1]
        },
        {
            "input": [1,1],
            "output": [0]
        }
    ]

    # Create a MLP with 2 input, a hidden layer with 2 nodes a single output node
    nn = Mlp(2,2,1)

    print("Training the network...")
    for i in range(10000):
        data = random.choice(training)
        nn.train(data["input"], data["output"])

    for i in range(2):
        for j in range(2):
            prediction = nn.predict([i,j]).item((0,0))
            print("Predicting XOR between {} and {} gave {} and the real is {} (Output: {})"
                .format(i, j, prediction > .5, bool(i) ^ bool(j), prediction))

def ocr():
    print("Loading training data...")
    df = pd.read_csv("../datasets/ocr_train.csv")
    print("Loaded {} rows.".format(df.shape[0]))
    nn = Mlp(784, 50, 10, learning_rate=.1)

    print("Training the network...")
    for i in range(30000):
        data = df.sample(n=1)
        label = data["label"].tolist()[0]
        inputs = list(data.iloc[0,1:])
        outputs = [0] * 10
        outputs[label] = 1
        nn.train(inputs, outputs)

    print("Trained successfuly.")
    print("Loading testing data...")
    ts = pd.read_csv("../datasets/ocr_test.csv")
    print("Loaded {} rows.".format(ts.shape[0]))
    print("Testing...")
    for i in range(2):
        data = ts.sample(n=1)
        #label = data["label"].tolist()[0]
        inputs = list(data.iloc[0,0:])
        output = nn.predict(inputs)
        #print("\tI should be {}".format(label))
        print(output)

#ocr()
xor()
