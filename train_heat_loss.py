from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

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

train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=8,
    shuffle=False,
)

test_loader = DataLoader(
    test_set,
    batch_size=8,
    shuffle=False,
)

mean, std = data.get_statistics()

###############################################################
# MULTIPLE RUNS
###############################################################

runs = 3

mse_all = []
rmse_all = []
mae_all = []
grad_all = []

###############################################################
# LOSS FUNCTIONS
###############################################################

def masked_mse(pred, target, mask):
    """
    Computes MSE only on held-out pixels.
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
    Parameters
    ----------
    T_seq : (B,T,H,W)
        Sequence of high-resolution temperature fields.
    """

    dTdt = (
        T_seq[:, 2:] -
        T_seq[:, :-2]
    ) / (2 * dt)

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

    return (residual ** 2).mean()


###############################################################
# START OF EACH RUN
###############################################################

for run in range(runs):

    print(f"\n========== Run {run+1}/{runs} ==========\n")

    seed = 42 + run

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ###########################################################
    # MODEL
    ###########################################################

    model = VarCNN().to(device)

    ###########################################################
    # TRAINABLE PHYSICAL PARAMETER
    ###########################################################

    lambda_phys = 1e-4

    kappa_raw = torch.nn.Parameter(
        torch.tensor(
            0.1,
            device=device,
        )
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [kappa_raw]
    )

    best_val = float("inf")
    
    results=[]
    ###############################################################
# TRAINING (100 epochs)
###############################################################

    for epoch in range(100):

        ###########################################################
        # TRAIN
        ###########################################################

        model.train()

        train_loss = 0.0

        for xb, yb, mask, hist in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            hist = hist.to(device)

            #######################################################
            # FORWARD
            #######################################################

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

            # High-resolution sequence:
            # [T(t-2), T(t-1), T̂(t)]
            sequence = torch.cat(
                [
                    hist,
                    pred,
                ],
                dim=1,
            )

            kappa = F.softplus(kappa_raw)

            physics_loss = heat_residual(
                sequence,
                kappa=kappa,
            )

            #######################################################
            # TOTAL LOSS
            #######################################################

            loss = (
                data_loss
                + lambda_phys * physics_loss
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ###########################################################
        # VALIDATION
        ###########################################################

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for xb, yb, mask, hist in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)
                mask = mask.to(device)
                hist = hist.to(device)

                pred = model(xb)

                ###################################################
                # DATA LOSS
                ###################################################

                data_loss = masked_mse(
                    pred,
                    yb,
                    mask,
                )

                ###################################################
                # PHYSICS LOSS
                ###################################################

                sequence = torch.cat(
                    [
                        hist,
                        pred,
                    ],
                    dim=1,
                )

                kappa = F.softplus(kappa_raw)

                physics_loss = heat_residual(
                    sequence,
                    kappa=kappa,
                )

                ###################################################
                # TOTAL LOSS
                ###################################################

                loss = (
                    data_loss
                    + lambda_phys * physics_loss
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        ###########################################################
        # SAVE BEST MODEL
        ###########################################################

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "kappa": kappa_raw.detach(),
                },
                f"saved_models/heat/best_model_run{run}_{lambda_phys}.pt",
            )

        ###########################################################
        # LOGGING
        ###########################################################

        print(
            f"Run {run+1} | "
            f"Epoch {epoch+1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )
        
        ###############################################################
# LOAD BEST MODEL FOR THIS RUN
###############################################################

    ckpt = torch.load(
        f"saved_models/heat/best_model_run{run}_{lambda_phys}.pt",
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    kappa_raw.data.copy_(
        ckpt["kappa"]
    )

    ###############################################################
    # TEST
    ###############################################################

    model.eval()

    predictions = []
    targets = []

    test_loss = 0.0

    with torch.no_grad():

        for xb, yb, mask, hist in test_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            hist = hist.to(device)

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

    print(f"\nRun {run+1}")
    print(f"Masked Test MSE: {test_loss:.6f}")

    ###############################################################
    # DENORMALIZE
    ###############################################################

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    pred_real = predictions.numpy() * std + mean
    true_real = targets.numpy() * std + mean

    ###############################################################
    # METRICS
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

    ###############################################################
    # GRADIENT ERROR
    ###############################################################

    mask2d = test_set.mask.squeeze(0).numpy().astype(bool)

    pred = pred_real.squeeze(1)
    true = true_real.squeeze(1)

    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    true_dx = true[:, :, 1:] - true[:, :, :-1]

    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    true_dy = true[:, 1:, :] - true[:, :-1, :]

    mask_dx = mask2d[:, 1:] & mask2d[:, :-1]
    mask_dy = mask2d[1:, :] & mask2d[:-1, :]

    grad_x = np.abs(
        pred_dx - true_dx
    )[..., mask_dx]

    grad_y = np.abs(
        pred_dy - true_dy
    )[..., mask_dy]

    gradient_error = (
        grad_x.mean() +
        grad_y.mean()
    ) / 2

    print(
        f"Gradient Error: {gradient_error:.6f}"
    )

    ###############################################################
    # STORE METRICS
    ###############################################################

    mse_all.append(mse)
    rmse_all.append(rmse)
    mae_all.append(mae)
    grad_all.append(gradient_error)
    results.append(
        {
            "Run":run+1,
            "MSE":mse,
            "RMSE":rmse,
            "MAE":mae,
            "Gradient_Error":gradient_error,
            "Kappa":F.softplus(kappa_raw).item()
        }
    )

    ###############################################################
    # ERROR MAP
    ###############################################################

    abs_error = np.abs(
        pred_real - true_real
    )

    mean_abs_error = (
        abs_error.mean(axis=0)
        .squeeze()
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        mean_abs_error,
        origin="lower",
        cmap="hot",
    )

    plt.colorbar(
        label="Absolute Error (K)"
    )

    plt.title(
        f"Mean Absolute Error Map (Run {run+1})"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(
        f"results/heat/mean_absolute_error_run{run+1}_{lambda_phys}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

###############################################################
# AVERAGE OVER ALL RUNS
###############################################################

print("\n=====================================")
print("Average Results over 3 Runs")
print("=====================================")

print(
    f"MSE            : {np.mean(mse_all):.6f} ± {np.std(mse_all):.6f}"
)

print(
    f"RMSE           : {np.mean(rmse_all):.6f} ± {np.std(rmse_all):.6f}"
)

print(
    f"MAE            : {np.mean(mae_all):.6f} ± {np.std(mae_all):.6f}"
)

print(
    f"Gradient Error : {np.mean(grad_all):.6f} ± {np.std(grad_all):.6f}"
)

###############################################################
# SAVE RESULTS
###############################################################

df = pd.DataFrame(results)

mean_row = {
    "Run": "Mean",
    "MSE": df["MSE"].mean(),
    "RMSE": df["RMSE"].mean(),
    "MAE": df["MAE"].mean(),
    "Gradient_Error": df["Gradient_Error"].mean(),
    "Kappa": df["Kappa"].mean(),
}

std_row = {
    "Run": "Std",
    "MSE": df["MSE"].std(ddof=1),
    "RMSE": df["RMSE"].std(ddof=1),
    "MAE": df["MAE"].std(ddof=1),
    "Gradient_Error": df["Gradient_Error"].std(ddof=1),
    "Kappa": df["Kappa"].std(ddof=1),
}

df = pd.concat(
    [
        df,
        pd.DataFrame([mean_row]),
        pd.DataFrame([std_row]),
    ],
    ignore_index=True,
)

df.to_csv(
    f"results/heat/metrics_summary_{lambda_phys}.csv",
    index=False,
)

print(df)