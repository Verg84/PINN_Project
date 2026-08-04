from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

parser=argparse.ArgumentParser()

parser.add_argument(
    "--lambda_smooth",
    dest="lambda_smooth",
    type=float,
    default=0.0,
    help="Weight for physics loss",
)

parser.add_argument(
    "--downsample_factor",
    type=int,
    default=4,
    help="Spatial downsampling factor for the dataset",
)

args=parser.parse_args()

lambda_smooth=args.lambda_smooth
downsample_factor=args.downsample_factor
###############################################################
# DEVICE
###############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

###############################################################
# DATA
###############################################################

data = TempDataModule(
    "data/temp.nc",
    sequence_length=3,
    downsample_factor=downsample_factor,
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
# MASKED DATA LOSS
###############################################################

def masked_mse(pred, target, mask):

    error = (pred - target) ** 2
    error = error * mask

    return error.sum() / mask.sum()

###############################################################
# LAPLACIAN SMOOTHNESS LOSS
###############################################################

def laplacian_smoothness(
    T_seq,
    dx=1.0,
    dy=1.0,
):
    """
    Computes ||∇²T||²

    Parameters
    ----------
    T_seq : (B,C,H,W)
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

    return (lap ** 2).mean()

###############################################################
# TRAINING SETTINGS
###############################################################


runs = 3

mse_all = []
rmse_all = []
mae_all = []
grad_all = []

results = []

###############################################################
# MULTIPLE TRAINING RUNS
###############################################################

for run in range(runs):

    print(f"\n========== Run {run + 1}/{runs} ==========\n")

    seed = 42 + run

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    ###########################################################
    # MODEL
    ###########################################################

    model = VarCNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    best_val = float("inf")

    ###########################################################
    # TRAINING
    ###########################################################

    for epoch in range(100):

        model.train()

        train_loss = 0.0

        for xb, yb, mask, *_ in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)

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
            # SMOOTHNESS LOSS
            ###################################################

            smooth_loss = laplacian_smoothness(pred)

            ###################################################

            loss = (
                data_loss
                + lambda_smooth * smooth_loss
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

            for xb, yb, mask, *_ in val_loader:

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

                loss = (
                    data_loss
                    + lambda_smooth * smooth_loss
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        ###########################################################
        # SAVE BEST MODEL
        ###########################################################

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                f"saved_models/smooth/best_model_{lambda_smooth}_run{run}_{downsample_factor}.pt",
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
            f"saved_models/smooth/best_model_{lambda_smooth}_run{run}_{downsample_factor}.pt",
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

        for xb, yb, mask, *_ in test_loader:

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

    grad_x = np.abs(pred_dx - true_dx)[..., mask_dx]
    grad_y = np.abs(pred_dy - true_dy)[..., mask_dy]

    gradient_error = (grad_x.mean() + grad_y.mean()) / 2

    print(f"Gradient Error: {gradient_error:.6f}")
    
        ###############################################################
    # ABSOLUTE ERROR MAP
    ###############################################################

    abs_error = np.abs(pred_real - true_real)

    mean_abs_error = abs_error.mean(axis=0).squeeze()

    plt.figure(figsize=(8, 6))

    plt.imshow(
        mean_abs_error,
        origin="lower",
        cmap="hot",
    )

    plt.colorbar(label="Absolute Error (K)")

    plt.title(
        f"Mean Absolute Error Map\n"
        f"Run {run+1}"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(
        f"results/smooth/mean_absolute_error_"
        f"{lambda_smooth}_run{run+1}_{downsample_factor}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    ###############################################################
    # STORE METRICS
    ###############################################################

    mse_all.append(mse)
    rmse_all.append(rmse)
    mae_all.append(mae)
    grad_all.append(gradient_error)

    results.append(
        {
            "Run": run + 1,
            "Lambda": lambda_smooth,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Gradient Error": gradient_error,
        }
    )

###############################################################
# SUMMARY OVER ALL RUNS
###############################################################

print("\n========================================")
print(f"Smoothness Loss (λ = {lambda_smooth})")
print("Average over 3 runs")
print("========================================")

print(
    f"MSE            : "
    f"{np.mean(mse_all):.6f} ± {np.std(mse_all):.6f}"
)

print(
    f"RMSE           : "
    f"{np.mean(rmse_all):.6f} ± {np.std(rmse_all):.6f}"
)

print(
    f"MAE            : "
    f"{np.mean(mae_all):.6f} ± {np.std(mae_all):.6f}"
)

print(
    f"Gradient Error : "
    f"{np.mean(grad_all):.6f} ± {np.std(grad_all):.6f}"
)

###############################################################
# SAVE RESULTS
###############################################################

df = pd.DataFrame(results)

csv_file = (
    f"results/smooth/"
    f"smooth_lambda_{lambda_smooth}_{downsample_factor}.csv"
)

df.to_csv(
    csv_file,
    index=False,
)

print(f"\nResults saved to {csv_file}")

###############################################################
# OPTIONAL: PRINT TABLE
###############################################################

print("\nPer-run Results")
print(df)