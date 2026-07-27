from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
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
val_loader = DataLoader(val_set, batch_size=8)
test_loader = DataLoader(test_set, batch_size=8)

mean, std = data.get_statistics()

###############################################################
# MODEL
###############################################################

model = VarCNN().to(device)



###############################################################
# LOSS FUNCTIONS
###############################################################

def masked_mse(pred, target, mask):
    """
    Computes MSE only on held-out pixels (mask=1).
    """

    error = (pred - target) ** 2

    error = error * mask

    return error.sum() / mask.sum()


###############################################################
# HEAT EQUATION RESIDUAL
###############################################################

def heat_residual(
    T_seq,
    kappa,
    dt=1.0,
    dx=1.0,
    dy=1.0,
):
    """
    T_seq shape:
    (B,T,H,W)

    T >= 3
    """

    dTdt = (T_seq[:, 2:] - T_seq[:, :-2]) / (2 * dt)

    T = T_seq[:, 1:-1]

    d2x = (
        T[:, :, :, 2:]
        - 2 * T[:, :, :, 1:-1]
        + T[:, :, :, :-2]
    ) / dx**2

    d2y = (
        T[:, :, 2:, :]
        - 2 * T[:, :, 1:-1, :]
        + T[:, :, :-2, :]
    ) / dy**2

    lap = (
        d2x[:, :, 1:-1, :]
        + d2y[:, :, :, 1:-1]
    )

    dTdt = dTdt[:, :, 1:-1, 1:-1]

    residual = dTdt - kappa * lap

    return (residual**2).mean()


###############################################################
# TRAINING
###############################################################

lambda_phys = 1e-4
kappa_raw= torch.nn.Parameter(
    torch.tensor(0.1, device=device)
)

optimizer = torch.optim.Adam(
    list(model.parameters()) + [kappa_raw]
)


best_val = float("inf")

for epoch in range(100):

    ###########################################################
    # TRAIN
    ###########################################################

    model.train()

    train_loss = 0

    for xb, yb, mask in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)
        mask = mask.to(device)

        pred = model(xb)

        #######################################################
        # DATA LOSS
        #######################################################

        data_loss = masked_mse(
            pred,
            yb,
            mask,
        )

        #######################################################
        # PHYSICS LOSS
        #######################################################

        sequence = torch.cat(
            [
                xb[:, 1:],
                pred,
            ],
            dim=1,
        )

        
        
        kappa = F.softplus(kappa_raw)
        
        physics_loss = heat_residual(sequence,kappa=kappa)

        #######################################################

        loss = data_loss + lambda_phys * physics_loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    ###########################################################
    # VALIDATION
    ###########################################################

    model.eval()

    val_loss = 0

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

            sequence = torch.cat(
                [
                    xb[:, 1:],
                    pred,
                ],
                dim=1,
            )

            physics_loss = heat_residual(
                sequence,
                kappa=kappa,
            )

            loss = data_loss + lambda_phys * physics_loss

            val_loss += loss.item()

    val_loss /= len(val_loader)

    if val_loss < best_val:

        best_val = val_loss

        torch.save({
            "model": model.state_dict(),
            "kappa": kappa_raw.detach()
},          "saved_models/heat/best_model_1e_4.pt")

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.6f} | "
        f"Val {val_loss:.6f}"
    )

###############################################################
# LOAD BEST MODEL
###############################################################

ckpt = torch.load("saved_models/heat/best_model_1e_4.pt", map_location=device)

model.load_state_dict(ckpt["model"])
kappa_raw.data.copy_(ckpt["kappa"])

###############################################################
# TEST
###############################################################

model.eval()

predictions = []
targets = []

test_loss = 0

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
# DENORMALIZE
###############################################################

predictions = torch.cat(predictions)
targets = torch.cat(targets)

pred_real = predictions.numpy() * std + mean
true_real = targets.numpy() * std + mean

###############################################################
# METRICS ON HELD-OUT PIXELS ONLY
###############################################################

mask = test_set.mask.squeeze(0).numpy().astype(bool)

mask = np.broadcast_to(
    mask,
    pred_real.shape,
)

pred_eval = pred_real[mask]
true_eval = true_real[mask]

mse = np.mean(
    (pred_eval - true_eval) ** 2
)

rmse = np.sqrt(mse)

mae = np.mean(
    np.abs(pred_eval - true_eval)
)

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
    "results/heat/mean_absolute_error_1e_4.png",
    dpi=300,
    bbox_inches="tight",
)


plt.show()