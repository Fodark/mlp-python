from mlp import Mlp
import random

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
        print("Predicting XOR between {} and {} gave {} and the real is {}"
            .format(i, j, prediction > .5, bool(i) ^ bool(j)))
