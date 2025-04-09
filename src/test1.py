from CalcGrad import *
from utils import draw_graph, view_dot

# Verify the our output using torch
import torch
from os import system

system('cls')

a = Value(2.0 + 1.0)
b = Value(4.0)
c = Value(6.0)
d = Value(7.0)

x = (a**3/c) + d + 2*b

x.backward()

print("a.grad: " + str(a.grad))
print("b.grad: " + str(b.grad))
print("c.grad: " + str(c.grad))
print("d.grad: " + str(d.grad))


dot = draw_graph(x)
view_dot(dot)

print('\n' + '-'*3 + "torch results" + '-'*10)

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = torch.tensor(6.0, requires_grad=True)
d = torch.tensor(7.0, requires_grad=True)


x = (a**3/c) + d + 2*b

x.backward()

print("a.grad: " + str(float(a.grad)))
print("b.grad: " + str(float(b.grad)))
print("c.grad: " + str(float(c.grad)))
print("d.grad: " + str(float(d.grad)))


# Using network class.
xs = [2.0, 3.0, -1.0]            # inputs

N = Network([4, 3, 3, 1])        # this represents the layers

preds: list[Value] = N(xs)       # calling the Network object with the inputs

for pred in preds:
    pred.backward()

view_dot(draw_graph(preds[0]))
