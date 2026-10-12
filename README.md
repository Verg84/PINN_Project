# Temperature Prediction with a Residual CNN

These scripts build and train a PyTorch convolutional neural network to predict a temperature field from three input frames. All training scripts use the same `VarCNN` model; they differ mainly in the loss function.

## Contents

- [Project overview](#project-overview)
- [Building the network](#building-the-network)
- [Training scripts](#training-scripts)
- [Understanding the losses](#understanding-the-losses)
- [How training works](#how-training-works)
- [Running the scripts](#running-the-scripts)
- [Understanding the results](#understanding-the-results)

## Project overview

The goal is to learn a mapping from three temperature frames to one predicted temperature field. Each frame is a two-dimensional grid, and the network processes all three frames together as input channels.

The experiments compare three ways of training the same network:

- **Data loss only:** learn from the target temperature values.
- **Data loss with a heat-equation penalty:** also encourage the predicted field to follow a diffusion relationship.
- **Data loss with a smoothness penalty:** also discourage strong spatial curvature in the predicted field.

The extra losses are calculated during training. They do not add layers to the network.

## Building the network

The network is defined in `src/models/cnn_model.py`.

PyTorch represents the input as `(B, 3, H, W)` and the output as `(B, 1, H, W)`. Here, `B` is the number of samples in a batch, and `H` and `W` are the grid height and width. The three input channels represent the three frames; the single output channel contains the predicted temperature values.

```python
import torch
from src.models.cnn_model import VarCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VarCNN(
    in_channels=3,
    hidden_channels=64,
    num_blocks=4,
).to(device)

# Example: a batch of 8 samples, each containing 3 temperature frames.
x = torch.randn(8, 3, 32, 32, device=device)
prediction = model(x)  # Shape: (8, 1, 32, 32)
```

The network has three parts:

1. **Encoder:** a 3×3 convolution converts the 3 input channels into 64 feature channels, followed by batch normalization and ReLU.
2. **Residual backbone:** four residual blocks process the features. Each block contains two 3×3 convolutions with batch normalization. The input is added to the block output before a final ReLU, helping gradients flow during training.
3. **Decoder:** a 3×3 convolution reduces the channels from 64 to 32, followed by batch normalization and ReLU. A final convolution produces one temperature channel.

Every convolution uses padding of 1, so the spatial dimensions stay the same. There is no pooling or upsampling in the model. The final layer has no activation, allowing continuous temperature predictions. Convolution weights use Kaiming initialization.

For example, an input of shape `(8, 3, 32, 32)` passes through these stages:

```text
Input                 (8,  3, 32, 32)
Encoder               (8, 64, 32, 32)
Four residual blocks  (8, 64, 32, 32)
Decoder hidden layer  (8, 32, 32, 32)
Prediction            (8,  1, 32, 32)
```

A convolution learns local spatial patterns, such as differences between nearby temperatures. ReLU introduces nonlinear behavior, and batch normalization normalizes intermediate features using batch statistics during training. A residual block adds its original input back to the learned transformation: `output = ReLU(block(input) + input)`.

Change `hidden_channels` to adjust the network width and `num_blocks` to adjust its depth. If you change the number of input frames, also update `in_channels` and the dataset's `sequence_length`.

## Training scripts

| Script | Training objective |
| --- | --- |
| `train_baseline.py` | Masked mean squared error (MSE). |
| `train_example_0.py` | Masked MSE plus an adjustable heat-equation loss on `temp.nc`, without input downsampling. Set `--lambda-phys 0` for data-only training. |
| `train_heat_loss.py` | Masked MSE plus a weighted heat-equation residual. A positive diffusivity is learned using `softplus(kappa_raw)`. |
| `train_heat_loss_hourly_data.py` | Heat-equation training using `temp_hourly.nc` instead of `temp.nc`. |
| `train_smooth_loss.py` | Masked MSE plus a weighted squared Laplacian penalty on the prediction. |

Use the baseline as a reference for the other experiments. Use `train_example_0.py` to compare different heat-loss weights with inputs at their original resolution. The two heat-loss scripts support a dataset downsampling factor and differ in which dataset they load. The smoothness script tests a spatial penalty without the temporal heat-equation term.

## Understanding the losses

### Masked data loss

The data loss measures the squared difference between the prediction and target, using only pixels selected by the mask:

```text
data_loss = sum(mask × (prediction − target)²) / sum(mask)
```

A mask value of 1 includes a pixel, while 0 ignores it. Dividing by the mask sum gives the average error over selected pixels.

### Heat-equation loss

The heat-equation penalty measures how closely the temperature sequence follows:

```text
dT/dt = kappa × Laplacian(T)
physics_loss = mean((dT/dt − kappa × Laplacian(T))²)
total_loss = data_loss + lambda_phys × physics_loss
```

The scripts form a sequence from `prev`, the model prediction, and `nxt`. A central difference estimates the time derivative from the surrounding frames, while spatial second differences estimate the Laplacian of the prediction. The penalty is computed on interior pixels because the finite differences need neighboring grid points.

`kappa` controls the diffusion strength. The scripts learn a raw parameter and apply `softplus` to keep `kappa` positive. The difference calculations use `dt = dx = dy = 1` by default, so these are grid-unit calculations; interpreting the learned value in physical units requires appropriate spacing and temperature scaling.

### Smoothness loss

The smoothness penalty uses only the predicted field:

```text
smooth_loss = mean(Laplacian(prediction)²)
total_loss = data_loss + lambda_smooth × smooth_loss
```

It discourages strong local curvature. Unlike the heat-equation loss, it does not compare temperature changes across time or learn a diffusion parameter.

For either penalty, a weight of 0 makes the total loss equal to the data loss. Increasing the weight gives the penalty more influence; the useful value depends on the relative scales of the losses.

## How training works

The scripts follow the same general workflow:

1. Load the dataset through `TempDataModule` and split it into training, validation and test sets.
2. Build a new `VarCNN` and move it to the GPU if CUDA is available, otherwise the CPU.
3. Predict a temperature field for each training batch and calculate the selected loss.
4. Use backpropagation and Adam to update the model parameters. Heat-loss experiments also include the raw diffusion parameter in the optimizer.
5. Evaluate the validation loss after each epoch and save the best checkpoint.
6. Reload the best checkpoint and evaluate the test set.

The dataset loader handles data preparation and supplies the masks and, for heat-loss training, the surrounding frames. Its implementation determines the exact split and input preparation.

The basic training step is:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
prediction = model(xb)
loss = ((prediction - yb).square() * mask).sum() / mask.sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Here, `xb`, `yb`, and `mask` must be on the same device as the model. Pixels with mask value 1 contribute to the data loss. The mask must contain at least one selected pixel.

## Running the scripts

Run from the project root. The scripts require `dataset/tmp_dataset.py`, PyTorch, NumPy, Matplotlib, and pandas for the baseline, heat and smoothness scripts, plus any dependencies required by the dataset loader.

Place the required NetCDF datasets in the project's `data/` folder:

| Dataset | Used by |
| --- | --- |
| `data/temp.nc` | `train_baseline.py`, `train_example_0.py`, `train_heat_loss.py`, `train_smooth_loss.py` |
| `data/temp_hourly.nc` | `train_heat_loss_hourly_data.py` |

The hourly dataset is approximately **142.30 MB**, exceeding GitHub's **100 MB** limit for regular Git files. It is excluded from the repository and must be supplied locally before running the hourly experiments. Cloning the repository does not provide this file.

Keep this entry in `.gitignore` to avoid committing the hourly dataset:

```gitignore
data/temp_hourly.nc
```

The example, heat and smoothness scripts create their output directories automatically. Ensure `saved_models/baseline/` and `results/baseline/` exist before running the baseline script.

```bash
python train_baseline.py
python train_example_0.py --lambda-phys 0
python train_heat_loss.py --lambda_phys 0.1 --downsample_factor 4
python train_heat_loss_hourly_data.py --lambda_phys 0.1 --downsample_factor 4
python train_smooth_loss.py --lambda_smooth 0.1 --downsample_factor 4
```

The heat and smoothness scripts default to a loss weight of `0.0` and a downsampling factor of `4`. Use a positive loss weight to enable the extra penalty. The baseline and example scripts use inputs without downsampling.

All five scripts default to three runs, batches of 8 and 100 epochs per run. In `train_example_0.py`, `--lambda-phys` is required; `--runs`, `--epochs` and `--batch-size` can also be adjusted:

```bash
python train_example_0.py --lambda-phys 0.1 --runs 3 --epochs 100 --batch-size 8
```

Each script saves the model with the lowest validation loss, evaluates it, and generates error maps and metric summaries.

| Script | Output locations |
| --- | --- |
| Baseline | `saved_models/baseline/`, `results/baseline/` |
| Example 0 | `saved_models/example_0/lambda_<weight>/`, `results/example_0/lambda_<weight>/` |
| Heat loss | `saved_models/heat/temp_data/`, `results/heat/temp_data/` |
| Hourly heat loss | `results/heat/hourly_data/new_data/` (checkpoints and results) |
| Smoothness loss | `saved_models/smooth/`, `results/smooth/new_data/` |

## Understanding the results

The scripts convert predictions and targets back from normalized values using `value × std + mean` before calculating temperature-space metrics.

| Result | Meaning |
| --- | --- |
| MSE | Average squared prediction error over the selected test pixels. Large errors receive more weight. |
| RMSE | Square root of MSE, expressed in the same units as the temperature. |
| MAE | Average absolute prediction error, also in temperature units. |
| Gradient error | Difference between predicted and target spatial changes, using neighboring selected pixels. |
| Mean absolute error map | Average absolute error at each grid location across test samples. |

Lower error metrics indicate closer agreement with the targets. The error map helps locate regions where predictions are less accurate. It covers the full grid, whereas the reported masked metrics use selected pixels.

Multiple runs show how results vary with random initialization. Compare both the average error and its variation across runs. Heat-loss experiments also report the learned positive `kappa` value.
