import numpy as np
import torch
import pandas
import argparse
import os

from torch.utils.data import DataLoader

from dataset import FMDataset, eval_data_transform
from model import NekoNet
from device import device
from predict import predict

final_test_data = pandas.read_csv("./data/fashion-mnist_test.csv")

final_test_data = FMDataset(
    final_test_data.iloc[:, 1:],
    final_test_data,
    eval_data_transform
)

final_test_loader = DataLoader(
    final_test_data,
    batch_size=512,
    shuffle=False,
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dist/final_ema.pt")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()
    os.makedirs("dist", exist_ok=True)

    net = NekoNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))

    preds = predict(net, final_test_loader)

    labels = final_test_data.labels
    accuracy = (labels == preds).mean()
    print(f"Validation Accuracy: {accuracy:.4f}")