from CalcGrad import MLP, Value
from utils import is_same_dim


# Create A neural network.

neural_network = MLP([2, 2, 1])         # 2 input, 2 hidden, 1 output

# Create a dataset                    # XOR gate
X = [[0, 0], [0, 1], [1, 0], [1, 1]]  # pairs of inputs sence we have 2 input nodes
Y = [[0], [1], [1], [0]]              # single groups sense we have 1 output node


y_pred: list[Value] = []
for x in X:
    y_pred.append(neural_network(x))  # Getting random preductions. | Untrained MLP (model)

if not is_same_dim(y_pred, Y):
    raise ValueError("Dimension mismatch")


# Calculate the loss                  # (RS)2 / [(RS)2 + (RA)2]
def calc_loss(y_pred: list, y: list) -> float:
    loss = 0.0
    for y_pred_, y_ in zip(y_pred, y):
        loss += (y_pred_ - y_) ** 2
    return loss


print(calc_loss(y_pred, Y))