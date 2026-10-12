import argparse
import json
import os
import csv

from dataset.tmp_dataset import TempDataModule
from src.models.cnn_model import VarCNN
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


###############################################################
# COMMAND-LINE ARGUMENTS
###############################################################

parser = argparse.ArgumentParser(
    description="Train temperature reconstruction model with heat-equation loss."
)

parser.add_argument(
    "--lambda-phys",
    type=float,
    required=True,
    help="Weight applied to the heat-equation physics loss."
)

parser.add_argument(
    "--runs",
    type=int,
    default=3,
    help="Number of independent training runs. Default: 3"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of training epochs. Default: 100"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size. Default: 8"
)

args = parser.parse_args()

lambda_phys = args.lambda_phys
runs = args.runs
epochs = args.epochs
batch_size = args.batch_size


###############################################################
# SAFE STRING FOR FILENAMES
###############################################################

def lambda_to_string(value):
    """
    Convert lambda value into a filename-safe string.

    Examples:
        0       -> 0
        0.01    -> 0p01
        0.001   -> 0p001
        1e-4    -> 0p0001
    """
    return f"{value:.10g}".replace(".", "p").replace("-", "m")


lambda_tag = lambda_to_string(lambda_phys)


###############################################################
# OUTPUT DIRECTORIES
###############################################################

results_dir = os.path.join(
    "results",
    "example_0",
    f"lambda_{lambda_tag}"
)

model_dir = os.path.join(
    "saved_models",
    "example_0",
    f"lambda_{lambda_tag}"
)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


###############################################################
# DEVICE
###############################################################

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("Experiment 0")
print("==========================================")
print(f"Device       : {device}")
print(f"lambda_phys  : {lambda_phys}")
print(f"Runs         : {runs}")
print(f"Epochs       : {epochs}")
print(f"Batch size   : {batch_size}")
print(f"Results dir  : {results_dir}")
print(f"Model dir    : {model_dir}")
print("==========================================")


###############################################################
# DATA
###############################################################

data = TempDataModule(
    "data/temp.nc",
    sequence_length=3,
    downsample_factor=None,
)

train_set, val_set, test_set = data.split()

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
)

mean, std = data.get_statistics()


###############################################################
# METRIC STORAGE
###############################################################

mse_all = []
rmse_all = []
mae_all = []
grad_all = []
test_masked_mse_all = []
kappa_all = []

run_results = []


###############################################################
# LOSS FUNCTIONS
###############################################################

def masked_mse(pred, target, mask):
    """
    Compute MSE only at pixels selected by mask.
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
    Heat-equation residual.

    Parameters
    ----------
    T_seq : Tensor
        Shape (B, T, H, W).

    kappa : Tensor
        Thermal diffusivity parameter.

    dt : float
        Temporal spacing.

    dx : float
        Grid spacing in x direction.

    dy : float
        Grid spacing in y direction.
    """

    # Central finite difference in time
    dTdt = (
        T_seq[:, 2:]
        - T_seq[:, :-2]
    ) / (2 * dt)

    # Middle temporal frame
    T = T_seq[:, 1:-1]

    # Second spatial derivative in x
    d2x = (
        T[:, :, :, 2:]
        - 2 * T[:, :, :, 1:-1]
        + T[:, :, :, :-2]
    ) / dx**2

    # Second spatial derivative in y
    d2y = (
        T[:, :, 2:, :]
        - 2 * T[:, :, 1:-1, :]
        + T[:, :, :-2, :]
    ) / dy**2

    # Laplacian
    lap = (
        d2x[:, :, 1:-1, :]
        + d2y[:, :, :, 1:-1]
    )

    # Match dimensions
    dTdt = dTdt[:, :, 1:-1, 1:-1]

    residual = dTdt - kappa * lap

    return (residual ** 2).mean()


###############################################################
# RUN EXPERIMENTS
###############################################################

for run in range(runs):

    print(
        f"\n========== Run {run + 1}/{runs} "
        f"| lambda={lambda_phys} ==========\n"
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
            device=device
        )
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + [kappa_raw]
    )

    best_val = float("inf")


    ###########################################################
    # TRAINING
    ###########################################################

    for epoch in range(epochs):

        #######################################################
        # TRAIN
        #######################################################

        model.train()

        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0

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
            # PHYSICS LOSS
            ###################################################

            sequence = torch.cat(
                [prev, pred, nxt],
                dim=1
            )

            # Ensures kappa > 0
            kappa = F.softplus(kappa_raw)

            physics_loss = heat_residual(
                sequence,
                kappa=kappa,
            )


            ###################################################
            # TOTAL LOSS
            ###################################################

            weighted_physics_loss = (
                lambda_phys * physics_loss
            )

            loss = (
                data_loss
                + weighted_physics_loss
            )


            ###################################################
            # BACKPROPAGATION
            ###################################################

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            ###################################################
            # ACCUMULATE LOSSES
            ###################################################

            train_loss += loss.item()
            train_data_loss += data_loss.item()
            train_physics_loss += physics_loss.item()


        train_loss /= len(train_loader)
        train_data_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)


        #######################################################
        # VALIDATION
        #######################################################

        model.eval()

        val_loss = 0.0
        val_data_loss = 0.0
        val_physics_loss = 0.0

        with torch.no_grad():

            for xb, yb, mask, prev, nxt in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)
                mask = mask.to(device)
                prev = prev.to(device)
                nxt = nxt.to(device)

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
                # PHYSICS LOSS
                ###############################################

                sequence = torch.cat(
                    [prev, pred, nxt],
                    dim=1
                )

                kappa = F.softplus(kappa_raw)

                physics_loss = heat_residual(
                    sequence,
                    kappa=kappa,
                )


                ###############################################
                # TOTAL LOSS
                ###############################################

                loss = (
                    data_loss
                    + lambda_phys * physics_loss
                )

                val_loss += loss.item()
                val_data_loss += data_loss.item()
                val_physics_loss += physics_loss.item()


        val_loss /= len(val_loader)
        val_data_loss /= len(val_loader)
        val_physics_loss /= len(val_loader)


        #######################################################
        # SAVE BEST MODEL
        #######################################################

        if val_loss < best_val:

            best_val = val_loss

            model_path = os.path.join(
                model_dir,
                f"best_model_run{run + 1}.pt"
            )

            torch.save(
                {
                    "model": model.state_dict(),
                    "kappa": kappa_raw.detach(),
                    "lambda_phys": lambda_phys,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "best_val_loss": best_val,
                },
                model_path,
            )


        #######################################################
        # LOGGING
        #######################################################

        print(
            f"Run {run + 1} | "
            f"Epoch {epoch + 1:03d} | "
            f"Train {train_loss:.6f} | "
            f"Data {train_data_loss:.6f} | "
            f"Physics {train_physics_loss:.6f} | "
            f"Weighted Physics "
            f"{lambda_phys * train_physics_loss:.6f} | "
            f"Val {val_loss:.6f}"
        )


    ###############################################################
    # LOAD BEST MODEL
    ###############################################################

    model_path = os.path.join(
        model_dir,
        f"best_model_run{run + 1}.pt"
    )

    ckpt = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model"]
    )

    kappa_raw.data.copy_(
        ckpt["kappa"]
    )

    learned_kappa = (
        F.softplus(kappa_raw)
        .detach()
        .cpu()
        .item()
    )


    ###############################################################
    # TEST
    ###############################################################

    model.eval()

    predictions = []
    targets = []

    test_loss = 0.0

    with torch.no_grad():

        for xb, yb, mask, prev, nxt in test_loader:

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

            predictions.append(
                pred.cpu()
            )

            targets.append(
                yb.cpu()
            )


    test_loss /= len(test_loader)

    print(
        f"\nRun {run + 1}"
    )

    print(
        f"Masked Test MSE: {test_loss:.6f}"
    )

    print(
        f"Learned kappa: {learned_kappa:.6f}"
    )


    ###############################################################
    # DENORMALIZE
    ###############################################################

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    pred_real = (
        predictions.numpy() * std + mean
    )

    true_real = (
        targets.numpy() * std + mean
    )


    ###############################################################
    # HELD-OUT PIXEL METRICS
    ###############################################################

    mask = (
        test_set.mask
        .squeeze(0)
        .numpy()
        .astype(bool)
    )

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
    print(f"MSE            : {mse:.6f}")
    print(f"RMSE           : {rmse:.6f}")
    print(f"MAE            : {mae:.6f}")


    ###############################################################
    # GRADIENT ERROR
    ###############################################################

    mask2d = (
        test_set.mask
        .squeeze(0)
        .numpy()
        .astype(bool)
    )

    pred_np = pred_real.squeeze(1)
    true_np = true_real.squeeze(1)

    pred_dx = (
        pred_np[:, :, 1:]
        - pred_np[:, :, :-1]
    )

    true_dx = (
        true_np[:, :, 1:]
        - true_np[:, :, :-1]
    )

    pred_dy = (
        pred_np[:, 1:, :]
        - pred_np[:, :-1, :]
    )

    true_dy = (
        true_np[:, 1:, :]
        - true_np[:, :-1, :]
    )

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
    # STORE METRICS
    ###############################################################

    mse_all.append(mse)
    rmse_all.append(rmse)
    mae_all.append(mae)
    grad_all.append(gradient_error)
    test_masked_mse_all.append(test_loss)
    kappa_all.append(learned_kappa)

    current_run_results = {
        "run": run + 1,
        "seed": seed,
        "lambda_phys": lambda_phys,
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "gradient_error": float(gradient_error),
        "test_masked_mse": float(test_loss),
        "kappa": float(learned_kappa),
        "best_val_loss": float(best_val),
    }

    run_results.append(
        current_run_results
    )


    ###############################################################
    # SAVE INDIVIDUAL RUN METRICS
    ###############################################################

    run_metrics_path = os.path.join(
        results_dir,
        f"metrics_run{run + 1}.json"
    )

    with open(
        run_metrics_path,
        "w"
    ) as f:

        json.dump(
            current_run_results,
            f,
            indent=4
        )


    ###############################################################
    # ERROR MAP
    ###############################################################

    abs_error = np.abs(
        pred_real - true_real
    )

    mean_abs_error = (
        abs_error
        .mean(axis=0)
        .squeeze()
    )

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
        f"Mean Absolute Error Map\n"
        f"Run {run + 1}, "
        f"$\\lambda_{{phys}}$={lambda_phys:g}"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    error_map_path = os.path.join(
        results_dir,
        f"mean_absolute_error_run{run + 1}"
        f"_lambda_{lambda_tag}.png"
    )

    plt.savefig(
        error_map_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved error map: {error_map_path}"
    )


###############################################################
# AVERAGE OVER ALL RUNS
###############################################################

summary = {
    "lambda_phys": lambda_phys,
    "runs": runs,

    "mse_mean": float(
        np.mean(mse_all)
    ),
    "mse_std": float(
        np.std(mse_all)
    ),

    "rmse_mean": float(
        np.mean(rmse_all)
    ),
    "rmse_std": float(
        np.std(rmse_all)
    ),

    "mae_mean": float(
        np.mean(mae_all)
    ),
    "mae_std": float(
        np.std(mae_all)
    ),

    "gradient_error_mean": float(
        np.mean(grad_all)
    ),
    "gradient_error_std": float(
        np.std(grad_all)
    ),

    "test_masked_mse_mean": float(
        np.mean(test_masked_mse_all)
    ),
    "test_masked_mse_std": float(
        np.std(test_masked_mse_all)
    ),

    "kappa_mean": float(
        np.mean(kappa_all)
    ),
    "kappa_std": float(
        np.std(kappa_all)
    ),
}


###############################################################
# PRINT SUMMARY
###############################################################

print("\n=====================================")
print(
    f"Average Results over {runs} Runs"
)
print(
    f"lambda_phys = {lambda_phys}"
)
print("=====================================")

print(
    f"MSE            : "
    f"{summary['mse_mean']:.6f} ± "
    f"{summary['mse_std']:.6f}"
)

print(
    f"RMSE           : "
    f"{summary['rmse_mean']:.6f} ± "
    f"{summary['rmse_std']:.6f}"
)

print(
    f"MAE            : "
    f"{summary['mae_mean']:.6f} ± "
    f"{summary['mae_std']:.6f}"
)

print(
    f"Gradient Error : "
    f"{summary['gradient_error_mean']:.6f} ± "
    f"{summary['gradient_error_std']:.6f}"
)

print(
    f"Masked Test MSE: "
    f"{summary['test_masked_mse_mean']:.6f} ± "
    f"{summary['test_masked_mse_std']:.6f}"
)

print(
    f"Kappa          : "
    f"{summary['kappa_mean']:.6f} ± "
    f"{summary['kappa_std']:.6f}"
)


###############################################################
# SAVE SUMMARY AS JSON
###############################################################

summary_json_path = os.path.join(
    results_dir,
    "summary.json"
)

with open(
    summary_json_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


###############################################################
# SAVE ALL RUNS AS CSV
###############################################################

runs_csv_path = os.path.join(
    results_dir,
    "all_runs.csv"
)

with open(
    runs_csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=run_results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        run_results
    )


###############################################################
# SAVE SUMMARY AS CSV
###############################################################

summary_csv_path = os.path.join(
    results_dir,
    "summary.csv"
)

with open(
    summary_csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary.keys()
    )

    writer.writeheader()

    writer.writerow(
        summary
    )


print("\n=====================================")
print("Results saved successfully")
print("=====================================")
print(f"JSON summary : {summary_json_path}")
print(f"CSV summary  : {summary_csv_path}")
print(f"All runs CSV : {runs_csv_path}")
print(f"Error maps   : {results_dir}")
print(f"Models       : {model_dir}")