# BackPropX

A scalar-valued **autograd engine** built from scratch in Python, with beautiful neural network visualizations powered by [Manim](https://www.manim.community/).

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Autograd Engine**: Automatic differentiation for scalar values with full backpropagation support
- **Neural Network Components**: Build MLPs with customizable architectures
- **Computation Graph Visualization**: Render computation graphs using Graphviz
- **Network Architecture Visualization**: Animate neural networks with Manim

## Installation

```bash
# Clone the repository
git clone https://github.com/Pkson13/BackpropX.git
cd BackPropX

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install the package
pip install .

```

## Quick Start

### Visualizing Neural Network Architecture

The quickest way to see BackPropX in action is to run the neural network visualization:

```bash
python BackpropX/visualize_nn.py --nin 3 --nouts 3 3 4 3 --quality medium_quality
```

**Arguments:**

- `--nin` — Number of input neurons
- `--nouts` — List of neurons per layer (e.g., `3 3 4 3` creates 4 layers)
- `--quality` — Rendering quality: `low_quality`, `medium_quality`, `high_quality`, or `fourk_quality`

**Example Output:**

https://github.com/user-attachments/assets/ef911eb2-9cd3-40c9-9e1f-b1e7c5e6a477

<!-- To add your video:
1. Open a GitHub issue in your repo
2. Drag & drop media/videos/720p30/MLPscene.mp4 into the issue
3. Copy the generated URL and replace the placeholder above
4. Delete the issue -->

### Using the Autograd Engine

```python
from BackpropX.engine import Value

# Create values
a = Value(2.0, label='a')
b = Value(3.0, label='b')

# Perform operations
c = a * b + a
c.label = 'c'

# Compute gradients
c.backward()

print(f"a.grad = {a.grad}")  # dc/da
print(f"b.grad = {b.grad}")  # dc/db
```

### Building a Neural Network

```python
from BackpropX.nn import MLP

# Create an MLP with 3 inputs and layers of [4, 4, 1] neurons
model = MLP(3, [4, 4, 1])

# Forward pass
x = [2.0, 3.0, -1.0]
output = model(x)

# Backpropagation
output.backward()

# Access all parameters
params = model.parameters()
```

### Visualizing the Computation Graph

```python
from BackpropX.engine import Value
from BackpropX.draw import draw_dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b
c.label = 'c'
c.backward()

# Renders and opens an SVG visualization
draw_dot(c)
```

### Visualizing Neural Network Architecture

See the [Quick Start](#quick-start) section above for full details on running the visualization.

## Project Structure

```
BackPropX_autograd/
├── BackpropX/
│   ├── __init__.py
│   ├── engine.py        # Core autograd Value class
│   ├── nn.py            # Neural network components (Neuron, Layer, MLP)
│   ├── draw.py          # Graphviz computation graph visualization
│   └── visualize_nn.py  # Manim neural network animation
├── tests/
│   ├── test_engine.py
│   └── test_neuralNet.py
├── output/              # Generated graph visualizations
├── media/               # Manim video outputs
├── pyproject.toml
└── README.md
```

## API Reference

### `Value` (engine.py)

| Method                          | Description                                  |
| ------------------------------- | -------------------------------------------- |
| `__add__`, `__mul__`, `__sub__` | Arithmetic operations with gradient tracking |
| `tanh()`                        | Hyperbolic tangent activation                |
| `backward()`                    | Compute gradients via reverse-mode autodiff  |

### Neural Network (nn.py)

| Class              | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `Neuron(nin)`      | Single neuron with `nin` inputs                             |
| `Layer(nin, nout)` | Layer with `nout` neurons, each having `nin` inputs         |
| `MLP(nin, nouts)`  | Multi-layer perceptron with architecture defined by `nouts` |

## Example: Training Loop

```python
from BackpropX.nn import MLP

# Create model
model = MLP(3, [4, 4, 1])

# Training data
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop
learning_rate = 0.1
for epoch in range(100):
    # Forward pass
    predictions = [model(x) for x in xs]
    loss = sum((pred - y) ** 2 for pred, y in zip(predictions, ys))

    # Zero gradients
    for p in model.parameters():
        p.grad = 0

    # Backward pass
    loss.backward()

    # Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

## License

MIT License - feel free to use this for learning and experimentation!
