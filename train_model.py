import os
import sys
from pathlib import Path

from modules.calibration import calibrate_young
from modules.pytorch_mlp import PytorchMLPReg
from modules.utils import load_dataset


def train_pytorch_model(dataset_path, from_real=False):
    x_train, y_train, x_test, y_test = load_dataset(dataset_path, from_real)

    mlp = PytorchMLPReg()

    mlp.train(x_train, y_train, x_test, y_test, n_epochs=2_000)

    dataset_fname = dataset_path.parts[-1].strip(".csv")
    fname = f"data/results/{dataset_fname}.pth"
    mlp.save(fname)
    print(f"Trained model saved at {fname}")


def calibrate_model(dataset_path, from_real=False):
    # use the dataset to calbirate for Young modulus

    # finite-diffference gradient descent
    E = calibrate_young(dataset_path, from_real)

    # save the calbiration to a file
    fname = "data/results/model_calibrated.json"
    data = {"young_modulus": E}
    raise NotImplementedError("write data to a json file")
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model using dataset")

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["pytorch", "calibrated"],
        default="pytorch",
        help="Model type: pytorch or calibrated",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/results/blueleg_beam_sphere.csv"),
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "-r",
        "--from-real",
        action="store_true",
        help="Use real-world dataset instead of synthetic",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    learn_from_real = args.from_real
    model_type = args.model_type
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    if model_type == "calibrated":
        calibrate_model(dataset_path, learn_from_real)
    elif model_type == "pytorch":
        train_pytorch_model(dataset_path, learn_from_real)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
