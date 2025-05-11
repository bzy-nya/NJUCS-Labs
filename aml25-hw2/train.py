import torch, torch.nn.functional as F, numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
import argparse, os, random

from sklearn.metrics import f1_score,accuracy_score

from dataset import train_labeled_data, train_unlabeled_data, valid_labeled_data
from model   import NekoNet
from device  import device, autocast
from predict import predict

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train(net, epoch, tau, ramp):
    net.train()
    total_loss = 0.0
    num_batches = 0
    pbar = tqdm(
        zip(labeled_data_loader, unlabeled_data_loader), 
        total=min(len(labeled_data_loader), len(unlabeled_data_loader)),
        unit="batch"
    )
    for (x_labeled, y_labeled), (x_unlabeled, _) in pbar:
        x_labeled = x_labeled.to(device)
        y_labeled = y_labeled.to(device)
        x_unlabeled = x_unlabeled.to(device)

        # supervised
        with autocast:
            logits_labeled = net(x_labeled)
        L_sup = F.cross_entropy(logits_labeled, y_labeled)

        # unsupervised
        with autocast:
            logits_unlabeled = net(x_unlabeled)

        with torch.no_grad():
            pseudo_probs = torch.softmax(logits_unlabeled, dim=1)
        conf, y_hat = pseudo_probs.max(dim=1)
        mask = conf.ge(tau).float()

        if mask.sum() > 0:
            L_unsup = (F.cross_entropy(logits_unlabeled, y_hat, reduction='none') * mask).mean()
        else:
            L_unsup = torch.zeros(1, device=device).squeeze()

        lam = min(1.0, epoch / ramp)
        loss = L_sup + lam * L_unsup

        opt.zero_grad(); loss.backward(); opt.step()
        ema.update()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_description(f"Epoch {epoch} Training")

    sch.step()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.6f}")

def eval(net, epoch, loader):
    net.eval()
    preds = predict(net, loader, bar_desc=f"Epoch {epoch} Evaluating") 
    labels = valid_labeled_data.labels
    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')

    print(f"Epoch {epoch} Accuracy: {accuracy:.6f}, Macro F1: {macro_f1:.6f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlabeled_batch_size", type=int, default=256)
    ap.add_argument("--labeled_batch_size",  type=int, default=768)
    ap.add_argument("--lr",     type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--tau",    type=float, default=0.96)
    ap.add_argument("--ramp",   type=int, default=15)  
    ap.add_argument("--seed",   type=int, default=2025)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    os.makedirs("dist", exist_ok=True)

    set_seed(args.seed)

    labeled_data_loader   = DataLoader(train_labeled_data, batch_size=args.labeled_batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_workers, persistent_workers=True if args.num_workers > 0 else False)
    unlabeled_data_loader = DataLoader(train_unlabeled_data,  batch_size=args.unlabeled_batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_workers, persistent_workers=True if args.num_workers > 0 else False)
    valid_data_loader     = DataLoader(valid_labeled_data, batch_size=512, shuffle=False, 
                        num_workers=args.num_workers, persistent_workers=True if args.num_workers > 0 else False)

    net = NekoNet().to(device)
    ema = ExponentialMovingAverage(net.parameters(), decay=0.999)

    opt = Adam(net.parameters(), lr=args.lr, weight_decay=1e-4) # Change SGD to Adam
    sch = CosineAnnealingLR(opt, T_max=args.epochs)
    
    for epoch in range(args.epochs):
        train(net, epoch, args.tau, args.ramp)
        eval(net, epoch, valid_data_loader)
        if (epoch+1) % 10 == 0:
            torch.save(
                {"model": net.state_dict(), "ema":   ema.state_dict()},
                f"dist/e{epoch:02d}.pt"
            )

    ema.store(); ema.copy_to(net.parameters())
    torch.save(net.state_dict(), "dist/final_ema.pt")