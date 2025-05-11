import numpy as np
import torch
import argparse
import os
import tqdm

from torch.utils.data import DataLoader

from dataset import test_data, valid_labeled_data
from model import NekoNet
from device import device, autocast

def predict(net, loader, bar_desc="Predicting"): 
    net.eval()

    pbar = tqdm.tqdm(loader, total=len(loader), desc=bar_desc, unit="batch")

    preds = []
    with torch.no_grad():
        for x_batch in pbar: 
            images_only = x_batch[0] 
            if not isinstance(images_only, torch.Tensor): 
                images_only = x_batch 
            with autocast:
                logits = net(images_only.to(device))
            preds.append(torch.argmax(logits,1).cpu().numpy())
    return np.concatenate(preds)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dist/final_ema.pt")
    ap.add_argument("--out",  default="dist/12345678.txt")
    args = ap.parse_args()
    os.makedirs("dist", exist_ok=True)

    net = NekoNet().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device))

    valid_loader_main = DataLoader(valid_labeled_data, batch_size=512)
    preds = predict(net, valid_loader_main)

    labels = valid_labeled_data.labels
    accuracy = (labels == preds).mean()
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    test_loader_main = DataLoader(test_data, batch_size=512)
    preds = predict(net, test_loader_main)

    np.savetxt(args.out, preds, fmt="%d")
    print(f"Wrote {args.out}")