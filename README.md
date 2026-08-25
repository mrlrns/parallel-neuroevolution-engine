# 🧬 Massively Parallel Neuroevolution for Aquatic Soft-Body Locomotion

![Evolution Demo](link_to_your_gif_here.gif) 
*(Note: I will soon add a GIF showing the elite mutant swimming)*

## 🚀 Overview
This repository implements a custom, fully vectorized 2D physics engine built from scratch in PyTorch to simulate soft-body aquatic creatures. 

Instead of relying on standard loops or pre-built engines, this project leverages `torch.func.vmap` to achieve **massively parallel neural network evaluations**. By vectorizing both the physics simulation and the Brain (Multi-Layer Perceptron), the engine trains batches of hundreds of creatures simultaneously directly on the GPU / Apple Silicon (MPS), eliminating CPU-GPU transfer bottlenecks.

## ⚙️ Core Architecture & Physics
The environment is designed to study Embodied AI and morphological evolution in fluid dynamics.

### 1. Vectorized Mass-Spring-Damper System
Creatures are dynamically generated as graphs of nodes (masses) and edges (muscles/bones). The physical interactions are resolved using matrix operations for high-throughput batch processing.

### 2. Hydrodynamic Drag Simulation
To force the neural networks to learn realistic swimming patterns rather than exploiting simulation glitches, a custom directional drag model is applied to every bone segment:

F_drag = -k_water * d * (v · n) * n

Where `d` is the segment length, `v` is the mean velocity of the connected nodes, and `n` is the normal vector of the segment.

### 3. Neural Control & Actuation
Each creature is controlled by a PyTorch Neural Network taking relative node coordinates, velocities, and a rhythmic clock signal as inputs. The network outputs target muscle contractions, updating the resting length of the springs at each simulation step.

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- PyTorch (CUDA or MPS enabled for parallel batch training)
- Pygame & OpenCV (for visualization and video export only)

### Running the Training
To launch the headless evolutionary training loop across a batch of 100 creatures:
```bash
python train.py