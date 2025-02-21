# CalcGrad

CalcGrad is a lightweight Python library for tracking numerical operations and computing gradients using automatic differentiation. It provides a simple yet powerful framework for backpropagation, making it ideal for machine learning and optimization tasks.

## Features
- **Automatic Differentiation**: Compute gradients efficiently with backpropagation.
- **Custom Computational Graph**: Track operations for each computation.
- **Neural Network Support**: Includes basic implementations of neurons, layers, and multi-layer perceptrons (MLP).

## Installation
Currently, CalcGrad is not available via `pip`. You can use it by cloning the repository and importing it into your project:

```sh
# Clone the repository
$ git clone https://github.com/yourusername/calcgrad.git
$ cd calcgrad
```

Then, import the library in your Python code:

```python
from calcgrad import Value, Neuron, Layer, MLP
```

## Usage
### Basic Operations
CalcGrad allows you to create `Value` objects that support arithmetic operations while tracking gradients.

```python
from calcgrad import Value

x = Value(3.0)
y = Value(2.0)
z = x * y + x ** 2
z.backward()

print(f"z: {z.data}, dz/dx: {x.grad}, dz/dy: {y.grad}")
```

### Building a Neural Network
CalcGrad includes simple classes for neurons, layers, and multi-layer perceptrons (MLP).

```python
from calcgrad import MLP

mlp = MLP(3, [4, 2, 1])  # 3 input neurons, 2 hidden layers (4 and 2 neurons), and 1 output neuron
x = [Value(0.5), Value(-1.2), Value(0.8)]
output = mlp(x)
print(output)
```

## API Reference
### `Value` Class
- `Value(data)`: Creates a scalar value that supports arithmetic operations.
- Supports `+`, `-`, `*`, `/`, `**` operations while tracking gradients.
- `backward()`: Computes the gradients via backpropagation.

### `Neuron` Class
- `Neuron(wNumber)`: A single neuron with `wNumber` weights.
- `__call__(x)`: Performs a forward pass using the tanh activation function.

### `Layer` Class
- `Layer(nin, nout)`: A layer of neurons.
- `__call__(x)`: Passes input through the layer.

### `MLP` Class
- `MLP(nin, nout)`: A simple multi-layer perceptron (MLP).
- `__call__(x)`: Performs forward propagation through all layers.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License.

