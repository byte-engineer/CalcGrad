import math
import random
import enum
from typing import Union

import math

# Supported operations 
class OP(enum.Enum):
    plus = 0
    multiply = 1
    exponential = 2
    tanh = 3


# this class is the actual enigne of the library.
# the class will keep tracking effect of different math oprations on the main class data `self.data` which is a normal float.   
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
        visited: set[Value] = set()

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
                for child in node.children:   # This tow lines will be ignored if there is no childs. 
                    build_topo(child)
                topo.append(node)             # This Line will executed if there is not childs.

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
    
    def set_grad(self, grad: float|Value):
        for w in self.weights:
            w.grad = grad if isinstance(grad, Value) else Value(grad)
        
        self.bias = grad if isinstance(grad, Value) else Value(grad)



class Layer:
    def __init__(self, input_count: int, output_count: int, initlization: float|None=None): 
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

    def set_grad(self, grad: float|Value):
        for n in self.neurons:
            n.set_grad(grad)



class Network:
    def __init__(self, layers: list[int], initlization: float|None=None) -> list[Value]:                 # [2, 4, 3] -->  2 input, 4 hidden, 3 output
        """
        Create a layer of Neurons `Neuron()`
        ### parameters
        `layers`: length of layers .e.g `[1, 3, 4, 1]` network of one input layer, tow hidden layers and one output layer.\n
        `initlization`: Will initlize all the neurons with some float value if not None.
        """
        if sum([1 if i > 0 else 0 for i in layers]) < len(layers):
            raise RuntimeError("layers cannot be of size zero")

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


    def parameters(self) -> list[Value]:                     
        para: list[Value] = []
        for layer in self.layers:
            para.extend(layer.parameters())
    
        unique_para = []
        seen = set()  # Track unique object IDs
        for p in para:
            if id(p) not in seen:
                seen.add(id(p))
                unique_para.append(p)
    
        return unique_para


    def calc_loss(self, y_train: list[Value], preductions: list[Value]) -> Value:

        if len(y_train) != len(preductions):
            raise ValueError("length of y_train and preductions is NOT equal")
        
        return sum([(pred - tr)**2 for tr, pred in zip(y_train, preductions)])**0.5
        

    def set_grad(self, grad: float|Value):
        for l in self.layers:
            l.set_grad(grad)


if __name__ == '__main__':
    pass