# Temperature Prediction with a Residual CNN

These scripts build and train a PyTorch convolutional neural network to predict a temperature field from three input frames. All training scripts use the same `VarCNN` model; they differ mainly in the loss function.

## Building the network

The network is defined in `src/models/cnn_model.py`.

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

Change `hidden_channels` to adjust the network width and `num_blocks` to adjust its depth. If you change the number of input frames, also update `in_channels` and the dataset's `sequence_length`.

## Training scripts

| Script | Training objective |
| --- | --- |
| `train_baseline.py` | Masked mean squared error (MSE). |
| `train_example_0.py` | Three runs on hourly data with `lambda_phys = 0`, so only the data loss contributes. |
| `train_heat_loss.py` | Masked MSE plus a weighted heat-equation residual. A positive diffusivity is learned using `softplus(kappa_raw)`. |
| `train_smooth_loss.py` | Masked MSE plus a weighted squared Laplacian penalty on the prediction. |

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

Run from the project root. The scripts require `dataset/tmp_dataset.py`, PyTorch, NumPy, Matplotlib, and pandas for the heat and smoothness scripts, plus any dependencies required by the dataset loader.

Provide `data/temp.nc` for baseline and smoothness training, and `data/temp_hourly.nc` for the hourly experiments. Create the checkpoint and results directories used by each script before running.

```bash
python train_baseline.py
python train_example_0.py
python train_heat_loss.py --lambda_phys 0.1 --downsample_factor 4
python train_smooth_loss.py --lambda_smooth 0.1 --downsample_factor 4
```

Both command-line loss weights default to `0.0`; use a positive value to enable the extra penalty. Training uses batches of 8 for 100 epochs and saves the model with the lowest validation loss. The scripts then evaluate it and generate error maps; the heat and smoothness scripts also save CSV metrics.
