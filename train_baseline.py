from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    downsample_factor=None,       # Original high-resolution inputs
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
# MASKED MSE LOSS
###############################################################

def masked_mse(pred, target, mask):
    """
    mask = 1 -> supervised pixels
    mask = 0 -> ignored pixels
    """

    error = (pred - target) ** 2

    error = error * mask

    return error.sum() / mask.sum()


###############################################################
# MULTIPLE RUNS
###############################################################

runs = 3

# Store metrics from each run
mse_all = []
rmse_all = []
mae_all = []
grad_all = []

# Store per-run results for CSV
results = []

# Store mean absolute error map from each run
error_maps = []

###############################################################
# THREE INDEPENDENT TRAINING RUNS
###############################################################

for run in range(runs):

    print("\n" + "=" * 60)
    print(f"RUN {run + 1}/{runs}")
    print("=" * 60)

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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    ###########################################################
    # BEST VALIDATION LOSS
    ###########################################################

    best_val = float("inf")

    checkpoint_path = (
        f"saved_models/baseline/best_model_run{run + 1}.pt"
    )

    ###########################################################
    # TRAINING
    ###########################################################

    for epoch in range(100):

        #######################################################
        # TRAIN
        #######################################################

        model.train()

        train_loss = 0.0

        for xb, yb, mask, hist, *_ in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)

            # hist is available because the dataset returns
            # (X, y, mask, y_hist).
            # It is not needed by the baseline model.
            hist = hist.to(device)

            ###################################################
            # FORWARD
            ###################################################

            pred = model(xb)

            ###################################################
            # MASKED DATA LOSS
            ###################################################

            loss = masked_mse(
                pred,
                yb,
                mask,
            )

            ###################################################
            # OPTIMIZATION
            ###################################################

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        #######################################################
        # VALIDATION
        #######################################################

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for xb, yb, mask, hist,*_ in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)
                mask = mask.to(device)
                hist = hist.to(device)

                ################################################
                # FORWARD
                ################################################

                pred = model(xb)

                ################################################
                # MASKED VALIDATION LOSS
                ################################################

                loss = masked_mse(
                    pred,
                    yb,
                    mask,
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        #######################################################
        # SAVE BEST MODEL
        #######################################################

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

        #######################################################
        # LOGGING
        #######################################################

        print(
            f"Run {run + 1} | "
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )

    ############################################################
    # LOAD BEST MODEL FOR THIS RUN
    ############################################################

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    ############################################################
    # TEST
    ############################################################

    model.eval()

    test_loss = 0.0

    predictions = []
    targets = []

    with torch.no_grad():

        for xb, yb, mask, hist,*_ in test_loader:

            xb = xb.to(device)
            yb = yb.to(device)
            mask = mask.to(device)
            hist = hist.to(device)

            ####################################################
            # FORWARD
            ####################################################

            pred = model(xb)

            ####################################################
            # TEST LOSS
            ####################################################

            loss = masked_mse(
                pred,
                yb,
                mask,
            )

            test_loss += loss.item()

            ####################################################
            # STORE PREDICTIONS
            ####################################################

            predictions.append(
                pred.cpu()
            )

            targets.append(
                yb.cpu()
            )

    test_loss /= len(test_loader)

    print(
        f"\nRun {run + 1} "
        f"Masked Test MSE: {test_loss:.6f}"
    )

    ############################################################
    # DENORMALIZE
    ############################################################

    predictions = torch.cat(
        predictions
    )

    targets = torch.cat(
        targets
    )

    pred_real = (
        predictions.numpy() * std + mean
    )

    true_real = (
        targets.numpy() * std + mean
    )
    
    ###############################################################
# HELD-OUT PIXEL METRICS FOR THIS RUN
###############################################################

    mask = test_set.mask.squeeze(0).numpy().astype(bool)

    mask = np.broadcast_to(
        mask,
        pred_real.shape,
    )

    pred_eval = pred_real[mask]
    true_eval = true_real[mask]

    ############################################################
    # MSE
    ############################################################

    mse = np.mean(
        (pred_eval - true_eval) ** 2
    )

    ############################################################
    # RMSE
    ############################################################

    rmse = np.sqrt(mse)

    ############################################################
    # MAE
    ############################################################

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

    mask2d = (
        test_set.mask
        .squeeze(0)
        .numpy()
        .astype(bool)
    )

    pred = pred_real.squeeze(1)
    true = true_real.squeeze(1)

    ############################################################
    # X GRADIENT
    ############################################################

    pred_dx = (
        pred[:, :, 1:]
        - pred[:, :, :-1]
    )

    true_dx = (
        true[:, :, 1:]
        - true[:, :, :-1]
    )

    ############################################################
    # Y GRADIENT
    ############################################################

    pred_dy = (
        pred[:, 1:, :]
        - pred[:, :-1, :]
    )

    true_dy = (
        true[:, 1:, :]
        - true[:, :-1, :]
    )

    ############################################################
    # VALID GRADIENT PIXELS
    ############################################################

    mask_dx = (
        mask2d[:, 1:]
        & mask2d[:, :-1]
    )

    mask_dy = (
        mask2d[1:, :]
        & mask2d[:-1, :]
    )

    grad_x = np.abs(
        pred_dx - true_dx
    )[..., mask_dx]

    grad_y = np.abs(
        pred_dy - true_dy
    )[..., mask_dy]

    gradient_error = (
        grad_x.mean()
        + grad_y.mean()
    ) / 2

    print(
        f"Gradient Error : {gradient_error:.6f}"
    )


###############################################################
# STORE METRICS FOR THIS RUN
###############################################################

    mse_all.append(mse)
    rmse_all.append(rmse)
    mae_all.append(mae)
    grad_all.append(gradient_error)

    results.append(
        {
            "Run": run + 1,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Gradient_Error": gradient_error,
        }
    )


###############################################################
# ABSOLUTE ERROR MAP FOR THIS RUN
###############################################################

    abs_error = np.abs(
        pred_real - true_real
    )

    mean_abs_error = (
        abs_error
        .mean(axis=0)
        .squeeze()
    )

    # Store for averaging across runs
    error_maps.append(
        mean_abs_error
    )

    ############################################################
    # SAVE INDIVIDUAL RUN ERROR MAP
    ############################################################

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        mean_abs_error,
        origin="lower",
        cmap="hot",
    )

    plt.colorbar(
        label="Absolute Error (K)"
    )

    plt.title(
        f"Mean Absolute Error Map - Run {run + 1}"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.savefig(
        f"results/baseline/"
        f"mean_absolute_error_run{run + 1}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


###############################################################
# CREATE DATAFRAME WITH PER-RUN RESULTS
###############################################################

df = pd.DataFrame(
    results
)


###############################################################
# MEAN
###############################################################

mean_row = {
    "Run": "Mean",
    "MSE": df["MSE"].mean(),
    "RMSE": df["RMSE"].mean(),
    "MAE": df["MAE"].mean(),
    "Gradient_Error": df["Gradient_Error"].mean(),
}


###############################################################
# STANDARD DEVIATION
###############################################################

std_row = {
    "Run": "Std",
    "MSE": df["MSE"].std(ddof=1),
    "RMSE": df["RMSE"].std(ddof=1),
    "MAE": df["MAE"].std(ddof=1),
    "Gradient_Error": df["Gradient_Error"].std(ddof=1),
}


###############################################################
# ADD MEAN AND STD TO DATAFRAME
###############################################################

df = pd.concat(
    [
        df,
        pd.DataFrame([mean_row]),
        pd.DataFrame([std_row]),
    ],
    ignore_index=True,
)


###############################################################
# SAVE CSV
###############################################################

df.to_csv(
    "results/baseline/metrics_summary.csv",
    index=False,
)


###############################################################
# PRINT RESULTS
###############################################################

print("\n")
print("=" * 60)
print("RESULTS OVER 3 INDEPENDENT RUNS")
print("=" * 60)

print("\nPer-run results:")
print(df.iloc[:runs].to_string(index=False))

print("\nMean ± Standard Deviation:")
print("--------------------------------------")

print(
    f"MSE            : "
    f"{df.loc[df['Run'] == 'Mean', 'MSE'].iloc[0]:.6f} ± "
    f"{df.loc[df['Run'] == 'Std', 'MSE'].iloc[0]:.6f}"
)

print(
    f"RMSE           : "
    f"{df.loc[df['Run'] == 'Mean', 'RMSE'].iloc[0]:.6f} ± "
    f"{df.loc[df['Run'] == 'Std', 'RMSE'].iloc[0]:.6f}"
)

print(
    f"MAE            : "
    f"{df.loc[df['Run'] == 'Mean', 'MAE'].iloc[0]:.6f} ± "
    f"{df.loc[df['Run'] == 'Std', 'MAE'].iloc[0]:.6f}"
)

print(
    f"Gradient Error : "
    f"{df.loc[df['Run'] == 'Mean', 'Gradient_Error'].iloc[0]:.6f} ± "
    f"{df.loc[df['Run'] == 'Std', 'Gradient_Error'].iloc[0]:.6f}"
)

print("\nCSV saved to:")
print("results/baseline/metrics_summary.csv")


###############################################################
# AVERAGE ERROR MAP OVER 3 RUNS
###############################################################

error_maps = np.stack(
    error_maps,
    axis=0,
)

average_error_map = np.mean(
    error_maps,
    axis=0,
)


###############################################################
# SAVE AVERAGED ERROR MAP
###############################################################

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    average_error_map,
    origin="lower",
    cmap="hot",
)

plt.colorbar(
    label="Absolute Error (K)"
)

plt.title(
    "Mean Absolute Error Map - Average of 3 Runs"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.savefig(
    "results/baseline/"
    "mean_absolute_error_average.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


###############################################################
# FINAL SUMMARY
###############################################################

print("\n")
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"MAE            : "
    f"{np.mean(mae_all):.6f} ± "
    f"{np.std(mae_all, ddof=1):.6f}"
)

print(
    f"RMSE           : "
    f"{np.mean(rmse_all):.6f} ± "
    f"{np.std(rmse_all, ddof=1):.6f}"
)

print(
    f"MSE            : "
    f"{np.mean(mse_all):.6f} ± "
    f"{np.std(mse_all, ddof=1):.6f}"
)

print(
    f"Gradient Error : "
    f"{np.mean(grad_all):.6f} ± "
    f"{np.std(grad_all, ddof=1):.6f}"
)

print("\nAverage error map saved to:")
print(
    "results/baseline/"
    "mean_absolute_error_average.png"
)