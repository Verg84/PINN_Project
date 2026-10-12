from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os


###############################################################
# ARGUMENTS
###############################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    "--lambda_smooth",
    dest="lambda_smooth",
    type=float,
    default=0.0,
    help="Weight for smoothness loss",
)

parser.add_argument(
    "--downsample_factor",
    type=int,
    default=4,
    help="Spatial downsampling factor for the dataset",
)

args = parser.parse_args()

lambda_smooth = args.lambda_smooth
downsample_factor = args.downsample_factor


###############################################################
# OUTPUT DIRECTORIES
###############################################################

os.makedirs(
    "saved_models/smooth",
    exist_ok=True,
)

os.makedirs(
    "results/smooth/new_data",
    exist_ok=True,
)


###############################################################
# DEVICE
###############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


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


###############################################################
# NORMALIZATION STATISTICS
###############################################################

mean, std = data.get_statistics()

print(f"Mean: {mean}")
print(f"Std : {std}")


###############################################################
# MASKED DATA LOSS
###############################################################

def masked_mse(
    pred,
    target,
    mask,
):

    error = (
        pred - target
    ) ** 2

    error = (
        error * mask
    )

    return (
        error.sum()
        / mask.sum()
    )


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

    ###########################################################
    # SECOND DERIVATIVE X
    ###########################################################

    d2x = (
        T_seq[:, :, :, 2:]
        - 2 * T_seq[:, :, :, 1:-1]
        + T_seq[:, :, :, :-2]
    ) / dx**2


    ###########################################################
    # SECOND DERIVATIVE Y
    ###########################################################

    d2y = (
        T_seq[:, :, 2:, :]
        - 2 * T_seq[:, :, 1:-1, :]
        + T_seq[:, :, :-2, :]
    ) / dy**2


    ###########################################################
    # LAPLACIAN
    ###########################################################

    lap = (
        d2x[:, :, 1:-1, :]
        + d2y[:, :, :, 1:-1]
    )


    ###########################################################
    # SMOOTHNESS LOSS
    ###########################################################

    return (
        lap ** 2
    ).mean()


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
# ERROR MAP SETTINGS
#
# SAME SCALE FOR ALL RUNS:
#
# 0 K -> 12 K
###############################################################

ERROR_VMIN = 0.0
ERROR_VMAX = 12.0

ERROR_TICKS = np.arange(
    0,
    13,
    2,
)


###############################################################
# MULTIPLE TRAINING RUNS
###############################################################

for run in range(runs):

    print(
        f"\n========== "
        f"Run {run + 1}/{runs} "
        f"==========\n"
    )


    ###########################################################
    # RANDOM SEED
    ###########################################################

    seed = (
        42 + run
    )

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    ###########################################################
    # MODEL
    ###########################################################

    model = VarCNN().to(
        device
    )


    ###########################################################
    # OPTIMIZER
    ###########################################################

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )


    ###########################################################
    # BEST VALIDATION LOSS
    ###########################################################

    best_val = float(
        "inf"
    )


    ###########################################################
    # TRAINING
    ###########################################################

    for epoch in range(100):


        #######################################################
        # TRAIN MODE
        #######################################################

        model.train()

        train_loss = 0.0


        #######################################################
        # TRAINING LOOP
        #######################################################

        for (
            xb,
            yb,
            mask,
            *_
        ) in train_loader:

            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            mask = mask.to(
                device
            )


            ###################################################
            # FORWARD
            ###################################################

            pred = model(
                xb
            )


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

            smooth_loss = (
                laplacian_smoothness(
                    pred
                )
            )


            ###################################################
            # TOTAL LOSS
            ###################################################

            loss = (
                data_loss
                + lambda_smooth
                * smooth_loss
            )


            ###################################################
            # BACKPROPAGATION
            ###################################################

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            ###################################################
            # ACCUMULATE TRAIN LOSS
            ###################################################

            train_loss += (
                loss.item()
            )


        #######################################################
        # AVERAGE TRAIN LOSS
        #######################################################

        train_loss /= len(
            train_loader
        )


        ###########################################################
        # VALIDATION
        ###########################################################

        model.eval()

        val_loss = 0.0


        with torch.no_grad():

            for (
                xb,
                yb,
                mask,
                *_
            ) in val_loader:

                xb = xb.to(
                    device
                )

                yb = yb.to(
                    device
                )

                mask = mask.to(
                    device
                )


                ###############################################
                # FORWARD
                ###############################################

                pred = model(
                    xb
                )


                ###############################################
                # DATA LOSS
                ###############################################

                data_loss = masked_mse(
                    pred,
                    yb,
                    mask,
                )


                ###############################################
                # SMOOTHNESS LOSS
                ###############################################

                smooth_loss = (
                    laplacian_smoothness(
                        pred
                    )
                )


                ###############################################
                # TOTAL VALIDATION LOSS
                ###############################################

                loss = (
                    data_loss
                    + lambda_smooth
                    * smooth_loss
                )


                ###############################################
                # ACCUMULATE VALIDATION LOSS
                ###############################################

                val_loss += (
                    loss.item()
                )


        #######################################################
        # AVERAGE VALIDATION LOSS
        #######################################################

        val_loss /= len(
            val_loader
        )


        ###########################################################
        # SAVE BEST MODEL
        ###########################################################

        if val_loss < best_val:

            best_val = (
                val_loss
            )

            checkpoint_path = (
                f"saved_models/smooth/"
                f"best_model_"
                f"{lambda_smooth}_"
                f"run{run}_"
                f"{downsample_factor}.pt"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )


        ###########################################################
        # LOGGING
        ###########################################################

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )


    ###############################################################
    # LOAD BEST MODEL
    ###############################################################

    checkpoint_path = (
        f"saved_models/smooth/"
        f"best_model_"
        f"{lambda_smooth}_"
        f"run{run}_"
        f"{downsample_factor}.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
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

        for (
            xb,
            yb,
            mask,
            *_
        ) in test_loader:

            xb = xb.to(
                device
            )

            yb = yb.to(
                device
            )

            mask = mask.to(
                device
            )


            ###################################################
            # PREDICTION
            ###################################################

            pred = model(
                xb
            )


            ###################################################
            # TEST LOSS
            ###################################################

            loss = masked_mse(
                pred,
                yb,
                mask,
            )

            test_loss += (
                loss.item()
            )


            ###################################################
            # STORE
            ###################################################

            predictions.append(
                pred.cpu()
            )

            targets.append(
                yb.cpu()
            )


    ###############################################################
    # AVERAGE TEST LOSS
    ###############################################################

    test_loss /= len(
        test_loader
    )

    print(
        f"\nMasked Test MSE: "
        f"{test_loss:.6f}"
    )


    ###############################################################
    # CONCATENATE
    ###############################################################

    predictions = torch.cat(
        predictions,
        dim=0,
    )

    targets = torch.cat(
        targets,
        dim=0,
    )


    ###############################################################
    # DENORMALIZE
    ###############################################################

    pred_real = (
        predictions.numpy()
        * std
        + mean
    )

    true_real = (
        targets.numpy()
        * std
        + mean
    )


    ###############################################################
    # PREDICTION DIAGNOSTICS
    ###############################################################

    print(
        f"\nPrediction diagnostics — "
        f"Run {run + 1}"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Prediction min  : "
        f"{np.nanmin(pred_real):.4f} K"
    )

    print(
        f"Prediction max  : "
        f"{np.nanmax(pred_real):.4f} K"
    )

    print(
        f"Prediction mean : "
        f"{np.nanmean(pred_real):.4f} K"
    )

    print(
        f"Target min      : "
        f"{np.nanmin(true_real):.4f} K"
    )

    print(
        f"Target max      : "
        f"{np.nanmax(true_real):.4f} K"
    )

    print(
        f"Target mean     : "
        f"{np.nanmean(true_real):.4f} K"
    )


    ###############################################################
    # HELD-OUT MASK
    ###############################################################

    mask2d = (
        test_set.mask
        .squeeze(0)
        .numpy()
        .astype(bool)
    )


    ###############################################################
    # BROADCAST MASK
    ###############################################################

    mask_full = np.broadcast_to(
        mask2d,
        pred_real.shape,
    )


    ###############################################################
    # HELD-OUT PIXELS
    ###############################################################

    pred_eval = (
        pred_real[
            mask_full
        ]
    )

    true_eval = (
        true_real[
            mask_full
        ]
    )


    ###############################################################
    # MSE
    ###############################################################

    mse = np.mean(
        (
            pred_eval
            - true_eval
        ) ** 2
    )


    ###############################################################
    # RMSE
    ###############################################################

    rmse = np.sqrt(
        mse
    )


    ###############################################################
    # MAE
    ###############################################################

    mae = np.mean(
        np.abs(
            pred_eval
            - true_eval
        )
    )


    ###############################################################
    # PRINT METRICS
    ###############################################################

    print(
        "\nHeld-out pixel metrics"
    )

    print(
        "----------------------"
    )

    print(
        f"MSE  : "
        f"{mse:.6f}"
    )

    print(
        f"RMSE : "
        f"{rmse:.6f}"
    )

    print(
        f"MAE  : "
        f"{mae:.6f}"
    )


    ###############################################################
    # GRADIENT ERROR
    ###############################################################

    pred = (
        pred_real.squeeze(1)
    )

    true = (
        true_real.squeeze(1)
    )


    ###############################################################
    # X GRADIENT
    ###############################################################

    pred_dx = (
        pred[:, :, 1:]
        - pred[:, :, :-1]
    )

    true_dx = (
        true[:, :, 1:]
        - true[:, :, :-1]
    )


    ###############################################################
    # Y GRADIENT
    ###############################################################

    pred_dy = (
        pred[:, 1:, :]
        - pred[:, :-1, :]
    )

    true_dy = (
        true[:, 1:, :]
        - true[:, :-1, :]
    )


    ###############################################################
    # GRADIENT MASKS
    ###############################################################

    mask_dx = (
        mask2d[:, 1:]
        & mask2d[:, :-1]
    )

    mask_dy = (
        mask2d[1:, :]
        & mask2d[:-1, :]
    )


    ###############################################################
    # GRADIENT ERROR X
    ###############################################################

    grad_x = np.abs(
        pred_dx
        - true_dx
    )[..., mask_dx]


    ###############################################################
    # GRADIENT ERROR Y
    ###############################################################

    grad_y = np.abs(
        pred_dy
        - true_dy
    )[..., mask_dy]


    ###############################################################
    # MEAN GRADIENT ERROR
    ###############################################################

    gradient_error = (
        grad_x.mean()
        + grad_y.mean()
    ) / 2


    print(
        f"Gradient Error: "
        f"{gradient_error:.6f}"
    )


    ###############################################################
    # ABSOLUTE ERROR MAP
    ###############################################################

    abs_error = np.abs(
        pred_real
        - true_real
    )


    ###############################################################
    # MEAN ABSOLUTE ERROR ACROSS TEST SAMPLES
    #
    # Shape:
    # (H, W)
    ###############################################################

    mean_abs_error = (
        abs_error
        .mean(axis=0)
        .squeeze()
    )


    ###############################################################
    # ERROR MAP DIAGNOSTICS
    ###############################################################

    print(
        f"\nError map diagnostics — "
        f"Run {run + 1}"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Error map shape : "
        f"{mean_abs_error.shape}"
    )

    print(
        f"Error min       : "
        f"{np.nanmin(mean_abs_error):.4f} K"
    )

    print(
        f"Error max       : "
        f"{np.nanmax(mean_abs_error):.4f} K"
    )

    print(
        f"Error mean      : "
        f"{np.nanmean(mean_abs_error):.4f} K"
    )

    print(
        f"Error median    : "
        f"{np.nanmedian(mean_abs_error):.4f} K"
    )

    print(
        f"Error std       : "
        f"{np.nanstd(mean_abs_error):.4f} K"
    )

    print(
        f"Error p99       : "
        f"{np.nanpercentile(mean_abs_error, 99):.4f} K"
    )

    print(
        f"Maximum single "
        f"absolute error : "
        f"{np.nanmax(abs_error):.4f} K"
    )


    ###############################################################
    # FIND MAXIMUM MEAN ERROR PIXEL
    ###############################################################

    max_error_idx = np.unravel_index(
        np.nanargmax(
            mean_abs_error
        ),
        mean_abs_error.shape,
    )

    print(
        f"Maximum mean-error pixel : "
        f"{max_error_idx}"
    )

    print(
        f"Mean error at that pixel : "
        f"{mean_abs_error[max_error_idx]:.4f} K"
    )


    ###############################################################
    # FIND MAXIMUM SINGLE ERROR
    ###############################################################

    max_single_idx = np.unravel_index(
        np.nanargmax(
            abs_error
        ),
        abs_error.shape,
    )

    print(
        f"Largest single-error index : "
        f"{max_single_idx}"
    )

    print(
        f"Prediction at max error     : "
        f"{pred_real[max_single_idx]:.4f} K"
    )

    print(
        f"Target at max error         : "
        f"{true_real[max_single_idx]:.4f} K"
    )

    print(
        f"Absolute error              : "
        f"{abs_error[max_single_idx]:.4f} K"
    )


    ###############################################################
    # ERROR MAP
    #
    # FIXED SCALE:
    #
    # 0 -> 12 K
    ###############################################################

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )


    ###############################################################
    # DISPLAY ERROR MAP
    ###############################################################

    im = ax.imshow(
        mean_abs_error,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        vmin=ERROR_VMIN,
        vmax=ERROR_VMAX,
    )


    ###############################################################
    # COLORBAR
    #
    # 0, 2, 4, 6, 8, 10, 12 K
    ###############################################################

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        ticks=ERROR_TICKS,
    )

    cbar.set_label(
        "Mean Absolute Error (K)"
    )


    ###############################################################
    # TITLE
    ###############################################################

    ax.set_title(
        f"Mean Absolute Error Map\n"
        f"Run {run + 1}, "
        f"λ={lambda_smooth}"
    )


    ###############################################################
    # AXIS LABELS
    ###############################################################

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )


    ###############################################################
    # REMOVE GRID
    ###############################################################

    ax.grid(
        False
    )

    ax.minorticks_off()


    ###############################################################
    # LAYOUT
    ###############################################################

    fig.tight_layout()


    ###############################################################
    # SAVE ERROR MAP
    ###############################################################

    error_map_file = (
        f"results/smooth/new_data/"
        f"mean_absolute_error_"
        f"{lambda_smooth}_"
        f"run{run + 1}_"
        f"{downsample_factor}_"
        f"scale_0_12.png"
    )

    fig.savefig(
        error_map_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


    ###############################################################
    # PRINT SAVED LOCATION
    ###############################################################

    print(
        f"Run {run + 1} "
        f"error map saved to:"
    )

    print(
        error_map_file
    )


    ###############################################################
    # STORE METRICS
    ###############################################################

    mse_all.append(
        mse
    )

    rmse_all.append(
        rmse
    )

    mae_all.append(
        mae
    )

    grad_all.append(
        gradient_error
    )


    results.append(
        {
            "Run":
                run + 1,

            "Lambda":
                lambda_smooth,

            "Downsample_Factor":
                downsample_factor,

            "MSE":
                mse,

            "RMSE":
                rmse,

            "MAE":
                mae,

            "Gradient_Error":
                gradient_error,
        }
    )


###############################################################
# SUMMARY OVER ALL RUNS
###############################################################

print(
    "\n========================================"
)

print(
    f"Smoothness Loss "
    f"(λ = {lambda_smooth})"
)

print(
    "Average over 3 runs"
)

print(
    "========================================"
)


###############################################################
# MSE
###############################################################

print(
    f"MSE            : "
    f"{np.mean(mse_all):.6f} ± "
    f"{np.std(mse_all, ddof=1):.6f}"
)


###############################################################
# RMSE
###############################################################

print(
    f"RMSE           : "
    f"{np.mean(rmse_all):.6f} ± "
    f"{np.std(rmse_all, ddof=1):.6f}"
)


###############################################################
# MAE
###############################################################

print(
    f"MAE            : "
    f"{np.mean(mae_all):.6f} ± "
    f"{np.std(mae_all, ddof=1):.6f}"
)


###############################################################
# GRADIENT ERROR
###############################################################

print(
    f"Gradient Error : "
    f"{np.mean(grad_all):.6f} ± "
    f"{np.std(grad_all, ddof=1):.6f}"
)


###############################################################
# SAVE RESULTS
###############################################################

df = pd.DataFrame(
    results
)


###############################################################
# MEAN ROW
###############################################################

mean_row = {

    "Run":
        "Mean",

    "Lambda":
        lambda_smooth,

    "Downsample_Factor":
        downsample_factor,

    "MSE":
        df["MSE"].mean(),

    "RMSE":
        df["RMSE"].mean(),

    "MAE":
        df["MAE"].mean(),

    "Gradient_Error":
        df["Gradient_Error"].mean(),
}


###############################################################
# STD ROW
###############################################################

std_row = {

    "Run":
        "Std",

    "Lambda":
        lambda_smooth,

    "Downsample_Factor":
        downsample_factor,

    "MSE":
        df["MSE"].std(
            ddof=1
        ),

    "RMSE":
        df["RMSE"].std(
            ddof=1
        ),

    "MAE":
        df["MAE"].std(
            ddof=1
        ),

    "Gradient_Error":
        df["Gradient_Error"].std(
            ddof=1
        ),
}


###############################################################
# ADD SUMMARY ROWS
###############################################################

df = pd.concat(
    [
        df,

        pd.DataFrame(
            [mean_row]
        ),

        pd.DataFrame(
            [std_row]
        ),
    ],
    ignore_index=True,
)


###############################################################
# CSV FILE
###############################################################

csv_file = (
    f"results/smooth/new_data/"
    f"smooth_lambda_"
    f"{lambda_smooth}_"
    f"{downsample_factor}.csv"
)


###############################################################
# SAVE CSV
###############################################################

df.to_csv(
    csv_file,
    index=False,
)


###############################################################
# FINAL OUTPUT
###############################################################

print(
    f"\nResults saved to:"
)

print(
    csv_file
)


###############################################################
# PRINT TABLE
###############################################################

print(
    "\nPer-run Results"
)

print(
    df
)


###############################################################
# PRINT ERROR MAP LOCATIONS
###############################################################

print(
    "\nError maps saved as:"
)

for run in range(runs):

    print(
        f"results/smooth/new_data/"
        f"mean_absolute_error_"
        f"{lambda_smooth}_"
        f"run{run + 1}_"
        f"{downsample_factor}_"
        f"scale_0_12.png"
    )