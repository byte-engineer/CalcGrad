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
    def __init__(self, weights_count: int, initlization: None|float =None):
        """
        Create a single Neuron
        ### parameters
        `weights_count`: number of weights.\n
        `initlization`: Will initlize all the weights and bias with some float value if not None.
        """
        self.weights: list[Value]  = [Value(random.uniform(-1, 1)) if initlization is None else initlization  for _ in range(weights_count)]
        self.bias: Value           = Value(random.uniform(-1, 1))  if initlization is None else initlization

    def __call__(self, x: list[float|Value]) -> Value:                          # Forward pass on the call operator.
        WxX: list[Value] = [wi*xi for wi, xi in zip(self.weights, x)]           # All wieghts * inputs
        act = sum(WxX, self.bias)
        return act.tanh()                                                       # Normlize the Sum using tanh() | -1 <-> 1

    def __repr__(self):
        weights_str = ", ".join(f"{w.data:6.3f}" for w in self.weights)
        return f"Neuron(weights=[{weights_str}], bias={self.bias.data:6.3f})"
    
    def parameters(self) -> list[Value]:
        self.weights.append(self.bias)
        return self.weights



class Layer:                                                                                # Layer of neurons
    def __init__(self, input_count: int, output_count: int, initlization: float|None=None): #|> MLP of 2, 2 
        """
        Create a layer of Neurons `Neuron()`
        ### parameters
        `input_count`: number of layer inputs.\n
        `outputs_count`: number of layer outputs.\n
        `initlization`: Will initlize all the neurons with some float value if not None.
        """
        self.neurons: list[Neuron] = [Neuron(input_count, initlization) for _ in range(output_count)]     #|> It's has tow layers of 2 neurons
#                                                                                                         #|> each neuron has 2 weights and 1 bias
    def __call__(self, x: list[float|Value]) -> list[Value]:                                              #|> Great & easy, right?
        outs: list[Value] = [n(x) for n in self.neurons]                                                  #
        return outs

    def __repr__(self):
        return f"Layer(neurons={len(self.neurons)})"                                                      # Assuming Layer has a list of neurons

    def parameters(self) -> list[Value]:
        para: list[Value] = []
        for neuron in self.neurons:
            para.extend(neuron.parameters())
        return para



class Network:
    def __init__(self, layers: list[int], initlization: float|None=None) -> list[Value]:       # [2, 4, 3] -->  2 input, 4 hidden, 3 output
        self.layers: list[Layer] = [Layer(layers[i], layers[i+1], initlization) for i in range(len(layers)-1)]

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

    def parameters(self):
        para: list[Value] = []
        for layer in self.layers:
            para.extend(layer.parameters())
        return para
