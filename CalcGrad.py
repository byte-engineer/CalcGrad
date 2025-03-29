import math
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Union

import math

class Value:
    def __init__(self, data, children=(), op=''):
        self.data: float = data                 # # #   
        self.children: tuple = children
        self.op: str = op
        self.grad: float = 0.0
        self._backward: function = lambda: None
        self.ValueId: int = id(self)


    def __repr__(self):
        return f"Value({self.data:6.3f})"

    def __add__(self, other: Union[float, int, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other: Union[float, int, "Value"]) -> "Value":
        return self + other


    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union[float, int, "Value"]) -> "Value":
        return self + (-other)

    def __rsub__(self, other: Union[float, int, "Value"]) -> "Value":
        return other + (-self)


    def __mul__(self, other: Union[float, int, "Value"]) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Union[float, int, "Value"]) -> "Value":
        return self * other

    def __pow__(self, other: Union[float, int, "Value"]) -> "Value":
        assert isinstance(other, (float, int)), "Just 'float' and 'int' supported for now!"
        out = Value(self.data ** other, (self, ), '^')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other: Union[float, int, "Value"]) -> "Value":
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
        visited: set[Value] = {}

        def build_topo(node: Value):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        for node in reversed(topo):
            node._backward()


    def zero_grad(self):
        topo: list[Value] = []
        visited: set = set()

        def build_topo(node: Value):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        for node in reversed(topo):
            node.grad = 0.0



class Neuron:
    def __init__(self, weights_count: int):                            # Will create a neuron with random weights
        self.weights: list[Value]  = [Value(random.uniform(-1, 1)) for _ in range(weights_count)]
        self.bias: Value           = Value(random.uniform(-1, 1))

    def __call__(self, x: list[float|Value]) -> Value:                          # Forward pass on the call operator.
        WxX: list[Value] = [wi*xi for wi, xi in zip(self.weights, x)]        # All wieghts * inputs
        act = sum(WxX, self.bias)                                         
        return act.tanh()                                              # Normlize the Sum using tanh() | -1 <-> 1

    def __repr__(self):
        weights_str = ", ".join(f"{w.data:6.3f}" for w in self.weights)
        return f"Neuron(weights=[{weights_str}], bias={self.bias.data:6.3f})"


class Layer:                                                                              # Layer of neurons   |
    def __init__(self, input_count: int, output_count: int ):                             #     in    2   out  |> MLP of 2, 2 
        self.neurons: list[Neuron] = [Neuron(input_count) for _ in range(output_count)]   #  *---#   -#---*    |> It's has tow layers of 2 neurons
#                                                                                         #       \ /          |> each neuron has 2 weights and 1 bias
    def __call__(self, x: list[float|Value]) -> list[Value]:                              #       / \          |> Great & easy, right?
        outs: list[Value] = [n(x) for n in self.neurons]                                  #  *---#   -#---*    |
        return outs

    def __repr__(self):
        return f"Layer(neurons={len(self.neurons)})"  # Assuming Layer has a list of neurons


from graphviz import Digraph

class Network:
    def __init__(self, layers: list[int]) -> list[Value]:       # [2, 4, 3] -->  2 input, 4 hidden, 3 output
        self.layers: list[Layer] = [Layer(layers[i], layers[i+1]) for i in range(len(layers)-1)]

    def __call__(self, inputs: list[float|Value]) -> list[Value]:
        """
        Input list length must must match the the input layer of the neural network
        """
        if (len(inputs) != len(self.layers[0].neurons)):
            RuntimeError("Input list length must must match the the input layer of the neural network")

        x = inputs
        for layer in self.layers:                               # Subistute x in each layer sequencially.
            x: list[Value] = layer(x)
        return x
    
    def __repr__(self):
        layers_str = "\n  ".join(f"Layer {i}: {str(layer)}" for i, layer in enumerate(self.layers))
        return f"Network(\n  {layers_str}\n)"

    def visualize(self, filename="network"):
        dot = Digraph(format="svg")
        dot.attr(rankdir="LR")  # Left to Right layout

        # Add nodes for each layer
        for layer_idx, layer in enumerate(self.layers):
            with dot.subgraph() as sub:
                sub.attr(rank="same")  # Keep neurons in the same layer at the same level
                for neuron_idx, neuron in enumerate(layer.neurons):
                    node_name = f"L{layer_idx}_N{neuron_idx}"
                    sub.node(node_name, label="⚪")

        # Add edges (connections)
        for layer_idx in range(len(self.layers) - 1):
            for neuron_idx, neuron in enumerate(self.layers[layer_idx].neurons):
                for next_neuron_idx in range(len(self.layers[layer_idx + 1].neurons)):
                    dot.edge(f"L{layer_idx}_N{neuron_idx}", f"L{layer_idx+1}_N{next_neuron_idx}")

        # Render the graph
        dot.render(filename, view=False)
