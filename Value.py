import math
import numpy as np
import matplotlib.pyplot as plt
import random


import math

class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data # int 
        self.children = children
        self.op = op
        self.grad = 0.0
        self._backward = lambda: None
        self.ValueId = id(self)

    def __repr__(self):
        return f"Value(data: {self.data})"

    def __add__(self, othaer):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other


    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other: int | float):
        assert isinstance(other, (float, int)), "Just 'float' and 'int' supported for now!"
        out = Value(self.data ** other, (self, ), '^')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        exp2x = math.exp(2*x)
        t = (exp2x - 1) / (exp2x + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0  # Start gradient from output node
        topo = []
        visited = set()

        def build_topo(node: Value):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, wNumber):
        self.w = [Value(random.uniform(-1, 1)) for i in range(wNumber)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):                                         # Forward pass on the call operator.
        # w*x + b
        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        return act.tanh()


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs


class MLP:
    def __init__(self, nin, nout):
        size = [nin] + nout
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(nout))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x