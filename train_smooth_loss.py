from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

###############################################################
# DEVICE
###############################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################
# DATA
###############################################################

data = TempDataModule(
    "data/temp.nc",
    sequence_length=3,
    downsample_factor=4,
)

train_set, val_set, test_set = data.split()

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

mean, std = data.get_statistics()

###############################################################
# MODEL
###############################################################

model = VarCNN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)

###############################################################
# MASKED DATA LOSS
###############################################################

def masked_mse(pred, target, mask):
    error = (pred - target) ** 2
    error = error * mask
    return error.sum() / mask.sum()

###############################################################
# LAPLACIAN SMOOTHNESS LOSS
###############################################################

def laplacian_smoothness(T_seq, dx=1.0, dy=1.0):
    """
    Computes ||∇²T||²

    Parameters
    ----------
    T_seq : (B,T,H,W)
    """

    d2x = (
        T_seq[:, :, :, 2:]
        - 2 * T_seq[:, :, :, 1:-1]
        + T_seq[:, :, :, :-2]
    ) / dx**2

    d2y = (
        T_seq[:, :, 2:, :]
        - 2 * T_seq[:, :, 1:-1, :]
        + T_seq[:, :, :-2, :]
    ) / dy**2

    lap = (
        d2x[:, :, 1:-1, :]
        + d2y[:, :, :, 1:-1]
    )

    return (lap**2).mean()

###############################################################
# TRAINING
###############################################################

lambda_smooth = 0

best_val = float("inf")

for epoch in range(100):

    model.train()

    train_loss = 0.0

    for xb, yb, mask in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)

        pred = model(xb)

        ##########################
        # Data loss
        ##########################

        data_loss = masked_mse(
            pred,
            yb,
            mask,
        )

        ##########################
        # Smoothness loss
        ##########################

        smooth_loss = laplacian_smoothness(pred)

        ##########################

        loss = data_loss + lambda_smooth * smooth_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    ###########################################################

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for xb, yb, mask in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)

            pred = model(xb)

            data_loss = masked_mse(
                pred,
                yb,
                mask,
            )

            smooth_loss = laplacian_smoothness(pred)

            loss = data_loss + lambda_smooth * smooth_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)

    if val_loss < best_val:

        best_val = val_loss

        torch.save(
            model.state_dict(),
            "saved_models/smooth/best_model_0.pt",
        )

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f}"
    )

###############################################################
# LOAD BEST MODEL
###############################################################

model.load_state_dict(
    torch.load(
        "saved_models/smooth/best_model_0.pt",
        map_location=device,
    )
)

###############################################################
# TEST
###############################################################

model.eval()

predictions = []
targets = []

test_loss = 0.0

with torch.no_grad():

    for xb, yb, mask in test_loader:

        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)

        pred = model(xb)

        loss = masked_mse(
            pred,
            yb,
            mask,
        )

        test_loss += loss.item()

        predictions.append(pred.cpu())
        targets.append(yb.cpu())

test_loss /= len(test_loader)

print(f"\nMasked Test MSE: {test_loss:.6f}")

###############################################################
# METRICS
###############################################################

predictions = torch.cat(predictions)
targets = torch.cat(targets)

pred_real = predictions.numpy() * std + mean
true_real = targets.numpy() * std + mean

mask = test_set.mask.squeeze(0).numpy().astype(bool)

mask = np.broadcast_to(mask, pred_real.shape)

pred_eval = pred_real[mask]
true_eval = true_real[mask]

mse = np.mean((pred_eval - true_eval) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(pred_eval - true_eval))

print("\nHeld-out pixel metrics")
print("----------------------")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")

# Gradient Error on the held out pixels
mask2d = test_set.mask.squeeze(0).numpy().astype(bool)

pred = pred_real.squeeze(1)
true = true_real.squeeze(1)

pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
true_dx = true[:, :, 1:] - true[:, :, :-1]

pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
true_dy = true[:, 1:, :] - true[:, :-1, :]

mask_dx = mask2d[:, 1:] & mask2d[:, :-1]
mask_dy = mask2d[1:, :] & mask2d[:-1, :]

grad_x = np.abs(pred_dx - true_dx)[..., mask_dx]
grad_y = np.abs(pred_dy - true_dy)[..., mask_dy]

gradient_error = (grad_x.mean() + grad_y.mean()) / 2
print(f'Gradient Error:{gradient_error:.6f}')

# Absolute Error Map
abs_error = np.abs(pred_real - true_real)
# Average over all test samples
mean_abs_error = abs_error.mean(axis=0).squeeze()
plt.figure(figsize=(8,6))
plt.imshow(mean_abs_error, origin="lower", cmap="hot")
plt.colorbar(label="Absolute Error (K)")
plt.title("Mean Absolute Error Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig(
    "results/smooth/mean_absolute_error_0.png",
    dpi=300,
    bbox_inches="tight",
)


plt.show()