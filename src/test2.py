from CalcGrad import Network, Value
from utils import draw_graph, view_dot



# Using network class.
xs = [2.0, 3.0, -1.0]            # inputs

N = Network([4, 2, 2, 1])        # this represents the layers

preds: list[Value] = N(xs)       # calling the Network object with the inputs

for pred in preds:
    pred.backward()

view_dot(draw_graph(preds[0]))
