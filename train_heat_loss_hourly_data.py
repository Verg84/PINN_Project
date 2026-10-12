from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


###############################################################
# ARGUMENTS
###############################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    "--lambda_phys",
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

args = parser.parse_args()

lambda_phys = args.lambda_phys
downsample_factor = args.downsample_factor


###############################################################
# OUTPUT DIRECTORIES
###############################################################

os.makedirs(
    "saved_models/heat/hourly_data",
    exist_ok=True,
)

os.makedirs(
    "results/heat/hourly_data/new_data",
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
    "data/temp_hourly.nc",
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
# MULTIPLE RUNS
###############################################################

runs = 3

mse_all = []
rmse_all = []
mae_all = []
grad_all = []

results = []


###############################################################
# MASKED MSE
###############################################################

def masked_mse(pred, target, mask):
    """
    Compute MSE only on held-out pixels.
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
    Compute heat-equation residual.

    Parameters
    ----------
    T_seq : Tensor
        Shape: (B, T, H, W)

        Temporal sequence:

            T(t-1), T(t), T(t+1)

    kappa : Tensor
        Thermal diffusivity.

    dt : float
        Temporal spacing.

    dx : float
        Grid spacing in x direction.

    dy : float
        Grid spacing in y direction.
    """

    ###########################################################
    # TEMPORAL DERIVATIVE
    ###########################################################

    dTdt = (
        T_seq[:, 2:]
        - T_seq[:, :-2]
    ) / (2 * dt)


    ###########################################################
    # CENTER TIME
    ###########################################################

    T = T_seq[:, 1:-1]


    ###########################################################
    # SECOND DERIVATIVE IN X
    ###########################################################

    d2x = (
        T[:, :, :, 2:]
        - 2 * T[:, :, :, 1:-1]
        + T[:, :, :, :-2]
    ) / dx**2


    ###########################################################
    # SECOND DERIVATIVE IN Y
    ###########################################################

    d2y = (
        T[:, :, 2:, :]
        - 2 * T[:, :, 1:-1, :]
        + T[:, :, :-2, :]
    ) / dy**2


    ###########################################################
    # LAPLACIAN
    ###########################################################

    lap = (
        d2x[:, :, 1:-1, :]
        + d2y[:, :, :, 1:-1]
    )


    ###########################################################
    # MATCH INTERIOR GRID
    ###########################################################

    dTdt = dTdt[
        :,
        :,
        1:-1,
        1:-1,
    ]


    ###########################################################
    # RESIDUAL
    ###########################################################

    residual = (
        dTdt
        - kappa * lap
    )

    return (
        residual ** 2
    ).mean()


###############################################################
# RUN EXPERIMENTS
###############################################################

for run in range(runs):

    print(
        f"\n"
        f"============================================\n"
        f"Run {run + 1}/{runs}\n"
        f"============================================"
    )


    ###########################################################
    # RANDOM SEED
    ###########################################################

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

    kappa_raw = torch.nn.Parameter(
        torch.tensor(
            0.1,
            dtype=torch.float32,
            device=device,
        )
    )


    ###########################################################
    # OPTIMIZER
    ###########################################################

    optimizer = torch.optim.Adam(
        list(model.parameters())
        + [kappa_raw]
    )

    best_val = float("inf")


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
        # TRAINING BATCHES
        #######################################################

        for xb, yb, mask, prev, nxt in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)

            prev = prev.to(device)
            nxt = nxt.to(device)


            ###################################################
            # FORWARD
            ###################################################

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
            # HIGH-RESOLUTION TEMPORAL SEQUENCE
            ###################################################

            sequence = torch.cat(
                [
                    prev,
                    pred,
                    nxt,
                ],
                dim=1,
            )


            ###################################################
            # POSITIVE KAPPA
            ###################################################

            kappa = F.softplus(
                kappa_raw
            )


            ###################################################
            # PHYSICS LOSS
            ###################################################

            physics_loss = heat_residual(
                sequence,
                kappa=kappa,
            )


            ###################################################
            # TOTAL LOSS
            ###################################################

            loss = (
                data_loss
                + lambda_phys
                * physics_loss
            )


            ###################################################
            # BACKPROPAGATION
            ###################################################

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            ###################################################
            # ACCUMULATE LOSS
            ###################################################

            train_loss += loss.item()


        #######################################################
        # AVERAGE TRAIN LOSS
        #######################################################

        train_loss /= len(
            train_loader
        )


        #######################################################
        # VALIDATION
        #######################################################

        model.eval()

        val_loss = 0.0


        with torch.no_grad():

            for (
                xb,
                yb,
                mask,
                prev,
                nxt,
            ) in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)
                mask = mask.to(device)

                prev = prev.to(device)
                nxt = nxt.to(device)


                ###############################################
                # FORWARD
                ###############################################

                pred = model(xb)


                ###############################################
                # DATA LOSS
                ###############################################

                data_loss = masked_mse(
                    pred,
                    yb,
                    mask,
                )


                ###############################################
                # TEMPORAL SEQUENCE
                ###############################################

                sequence = torch.cat(
                    [
                        prev,
                        pred,
                        nxt,
                    ],
                    dim=1,
                )


                ###############################################
                # POSITIVE KAPPA
                ###############################################

                kappa = F.softplus(
                    kappa_raw
                )


                ###############################################
                # PHYSICS LOSS
                ###############################################

                physics_loss = heat_residual(
                    sequence,
                    kappa=kappa,
                )


                ###############################################
                # TOTAL VALIDATION LOSS
                ###############################################

                loss = (
                    data_loss
                    + lambda_phys
                    * physics_loss
                )

                val_loss += (
                    loss.item()
                )


        #######################################################
        # AVERAGE VALIDATION LOSS
        #######################################################

        val_loss /= len(
            val_loader
        )


        #######################################################
        # SAVE BEST MODEL
        #######################################################

        if val_loss < best_val:

            best_val = val_loss

            checkpoint_path = (
                "results/heat/hourly_data/new_data/"
                f"best_model_run{run}_"
                f"{lambda_phys}_"
                f"{downsample_factor}.pt"
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "kappa":
                        kappa_raw
                        .detach()
                        .clone(),
                },
                checkpoint_path,
            )


        #######################################################
        # LOGGING
        #######################################################

        print(
            f"Run {run + 1} | "
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"Kappa "
            f"{F.softplus(kappa_raw).item():.6f}"
        )


    ###############################################################
    # LOAD BEST MODEL
    ###############################################################

    checkpoint_path = (
        "results/heat/hourly_data/new_data/"
        f"best_model_run{run}_"
        f"{lambda_phys}_"
        f"{downsample_factor}.pt"
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )


    ###############################################################
    # LOAD MODEL PARAMETERS
    ###############################################################

    model.load_state_dict(
        ckpt["model"]
    )


    ###############################################################
    # LOAD KAPPA
    ###############################################################

    with torch.no_grad():

        kappa_raw.copy_(
            ckpt["kappa"].to(device)
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
            prev,
            nxt,
        ) in test_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)


            #######################################################
            # PREDICT
            #######################################################

            pred = model(xb)


            #######################################################
            # TEST LOSS
            #######################################################

            loss = masked_mse(
                pred,
                yb,
                mask,
            )

            test_loss += (
                loss.item()
            )


            #######################################################
            # STORE
            #######################################################

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
        f"\nRun {run + 1}"
    )

    print(
        f"Masked Test MSE: "
        f"{test_loss:.6f}"
    )


    ###############################################################
    # CONCATENATE TEST DATA
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
    # 2D HELD-OUT MASK
    ###############################################################

    mask2d = (
        test_set.mask
        .squeeze(0)
        .numpy()
        .astype(bool)
    )


    ###############################################################
    # BROADCAST MASK TO TEST DATA
    ###############################################################

    mask_full = np.broadcast_to(
        mask2d,
        pred_real.shape,
    )


    ###############################################################
    # HELD-OUT PIXELS ONLY
    ###############################################################

    pred_eval = pred_real[
        mask_full
    ]

    true_eval = true_real[
        mask_full
    ]


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
        f"MSE  : {mse:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"MAE  : {mae:.6f}"
    )


    ###############################################################
    # GRADIENT ERROR
    ###############################################################

    pred_spatial = (
        pred_real.squeeze(1)
    )

    true_spatial = (
        true_real.squeeze(1)
    )


    ###############################################################
    # X GRADIENT
    ###############################################################

    pred_dx = (
        pred_spatial[:, :, 1:]
        - pred_spatial[:, :, :-1]
    )

    true_dx = (
        true_spatial[:, :, 1:]
        - true_spatial[:, :, :-1]
    )


    ###############################################################
    # Y GRADIENT
    ###############################################################

    pred_dy = (
        pred_spatial[:, 1:, :]
        - pred_spatial[:, :-1, :]
    )

    true_dy = (
        true_spatial[:, 1:, :]
        - true_spatial[:, :-1, :]
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
    # GRADIENT DIFFERENCE
    ###############################################################

    grad_x = np.abs(
        pred_dx
        - true_dx
    )[:, mask_dx]

    grad_y = np.abs(
        pred_dy
        - true_dy
    )[:, mask_dy]


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
    # LEARNED KAPPA
    ###############################################################

    learned_kappa = (
        F.softplus(
            kappa_raw
        ).item()
    )

    print(
        f"Learned Kappa: "
        f"{learned_kappa:.6f}"
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
            "Run": run + 1,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Gradient_Error":
                gradient_error,
            "Kappa":
                learned_kappa,
        }
    )


    ###############################################################
    # ERROR MAP
    #
    # CREATE ONE ERROR MAP FOR EVERY RUN
    ###############################################################

    print(
        f"\nCreating error map for Run {run + 1}..."
    )


    ###########################################################
    # ABSOLUTE ERROR
    ###########################################################

    abs_error = np.abs(
        pred_real
        - true_real
    )


    ###########################################################
    # MEAN ABSOLUTE ERROR ACROSS TEST SAMPLES
    #
    # Shape:
    # (H, W)
    ###########################################################

    mean_abs_error = (
        abs_error
        .mean(axis=0)
        .squeeze()
    )


    ###########################################################
    # DIAGNOSTIC INFORMATION
    ###########################################################

    print(
        "Error map shape:",
        mean_abs_error.shape
    )

    print(
        "Error min:",
        np.nanmin(mean_abs_error)
    )

    print(
        "Error max:",
        np.nanmax(mean_abs_error)
    )

    print(
        "Error mean:",
        np.nanmean(mean_abs_error)
    )

    print(
        "Error median:",
        np.nanmedian(mean_abs_error)
    )

    print(
        "Error std:",
        np.nanstd(mean_abs_error)
    )


    ###########################################################
    # PREDICTION / TARGET RANGE
    ###########################################################

    print(
        "pred_real min/max:",
        np.nanmin(pred_real),
        np.nanmax(pred_real)
    )

    print(
        "true_real min/max:",
        np.nanmin(true_real),
        np.nanmax(true_real)
    )

    print(
        "Max absolute prediction difference:",
        np.nanmax(
            np.abs(
                pred_real
                - true_real
            )
        )
    )


    ###########################################################
    # FIND MAXIMUM MEAN ERROR PIXEL
    ###########################################################

    max_idx = np.unravel_index(
        np.nanargmax(mean_abs_error),
        mean_abs_error.shape,
    )

    print(
        "Maximum mean-error pixel:",
        max_idx
    )

    print(
        "Mean absolute error at that pixel:",
        mean_abs_error[max_idx]
    )


    ###########################################################
    # LARGEST SINGLE PREDICTION ERROR
    ###########################################################

    single_error = np.abs(
        pred_real
        - true_real
    )

    single_idx = np.unravel_index(
        np.nanargmax(single_error),
        single_error.shape,
    )

    print(
        "Largest single-error index:",
        single_idx
    )

    print(
        "Prediction at largest error:",
        pred_real[single_idx]
    )

    print(
        "Target at largest error:",
        true_real[single_idx]
    )

    print(
        "Absolute error:",
        single_error[single_idx]
    )


    ###########################################################
    # FIXED COLOR SCALE
    #
    # SAME SCALE FOR EVERY RUN:
    #
    # 0 K -> 12 K
    ###########################################################

    vmin = 0.0
    vmax = 12.0

    print(
        f"Plot color range: "
        f"{vmin:.1f} to {vmax:.1f} K"
    )


    ###########################################################
    # CREATE FIGURE
    ###########################################################

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )


    ###########################################################
    # DISPLAY ERROR MAP
    ###########################################################

    im = ax.imshow(
        mean_abs_error,
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )


    ###########################################################
    # COLORBAR
    #
    # 0, 2, 4, 6, 8, 10, 12 K
    ###########################################################

    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        ticks=np.arange(
            0,
            13,
            2,
        ),
    )

    cbar.set_label(
        "Mean Absolute Error (K)"
    )


    ###########################################################
    # TITLE
    ###########################################################

    ax.set_title(
        f"Mean Absolute Error Map (Run {run + 1})"
    )


    ###########################################################
    # AXIS LABELS
    ###########################################################

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )


    ###########################################################
    # REMOVE GRID
    ###########################################################

    ax.grid(False)

    ax.minorticks_off()


    ###########################################################
    # LAYOUT
    ###########################################################

    fig.tight_layout()


    ###########################################################
    # SAVE ERROR MAP
    ###########################################################

    error_map_path = (
        "results/heat/hourly_data/new_data/"
        f"mean_absolute_error_run{run + 1}_"
        f"{lambda_phys}_"
        f"{downsample_factor}.png"
    )

    fig.savefig(
        error_map_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    ###########################################################
    # PRINT SAVED LOCATION
    ###########################################################

    print(
        f"Run {run + 1} error map saved to:"
    )

    print(
        error_map_path
    )


###############################################################
# AVERAGE RESULTS OVER ALL RUNS
###############################################################

print(
    "\n====================================="
)

print(
    "Average Results over 3 Runs"
)

print(
    "====================================="
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
# CREATE RESULTS DATAFRAME
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

    "MSE":
        df["MSE"].mean(),

    "RMSE":
        df["RMSE"].mean(),

    "MAE":
        df["MAE"].mean(),

    "Gradient_Error":
        df["Gradient_Error"].mean(),

    "Kappa":
        df["Kappa"].mean(),
}


###############################################################
# STANDARD DEVIATION ROW
###############################################################

std_row = {

    "Run":
        "Std",

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

    "Kappa":
        df["Kappa"].std(
            ddof=1
        ),
}


###############################################################
# APPEND SUMMARY ROWS
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
# SAVE METRICS
###############################################################

csv_path = (
    "results/heat/hourly_data/new_data/"
    f"metrics_summary_"
    f"{lambda_phys}_"
    f"{downsample_factor}.csv"
)

df.to_csv(
    csv_path,
    index=False,
)


###############################################################
# PRINT FINAL RESULTS
###############################################################

print(
    "\nFinal results:"
)

print(
    df
)

print(
    f"\nMetrics saved to: "
    f"{csv_path}"
)