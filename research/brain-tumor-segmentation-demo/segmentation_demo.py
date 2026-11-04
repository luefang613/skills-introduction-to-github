"""Standalone PyTorch segmentation demo using *synthetic* image masks.

This is a newly constructed engineering smoke test, NOT a re-creation of
archived BraTS scores or a clinical brain-tumor detection system.
"""
from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def seed_all(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SyntheticSegmentation(Dataset):
    """Deterministic toy image data: noisy disks and rectangles, labels 0/1/2."""
    def __init__(self, count: int = 24, size: int = 32, seed: int = 2026):
        if size < 16 or size % 4:
            raise ValueError('size must be >=16 and divisible by 4')
        self.count, self.size, self.seed = count, size, seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx)
        n = self.size
        yy, xx = np.mgrid[:n, :n]
        cx, cy = rng.integers(n // 4, 3 * n // 4, size=2)
        r = rng.integers(max(2, n // 8), max(3, n // 5))
        mask = np.zeros((n,n),dtype=np.int64)
        mask[(xx-cx)**2 + (yy-cy)**2 <= r*r] = 1
        x1, y1 = rng.integers(1, n // 3,size=2)
        mask[y1:y1+n//5,x1:x1+n//5] = 2
        signal = mask.astype(np.float32)/2
        image = signal + rng.normal(0,0.08,(n,n)).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.ReLU(),nn.Conv2d(cout,cout,3,padding=1),nn.ReLU())
    def forward(self,x): return self.layers(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 3, width: int = 8):
        super().__init__()
        self.enc1 = ConvBlock(in_channels,width)
        self.enc2 = ConvBlock(width,width*2)
        self.mid = ConvBlock(width*2,width*4)
        self.dec2 = ConvBlock(width*6,width*2)
        self.dec1 = ConvBlock(width*3,width)
        self.head = nn.Conv2d(width,num_classes,1)

    def forward(self,x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a,2))
        c = self.mid(F.max_pool2d(b,2))
        c = F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False)
        c = self.dec2(torch.cat([c,b],dim=1))
        c = F.interpolate(c,size=a.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(self.dec1(torch.cat([c,a],dim=1)))


@torch.no_grad()
def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, nclasses: int = 3) -> dict[str,float]:
    pred = logits.argmax(1)
    dices,ious = [],[]
    for label in range(nclasses):
        p = (pred==label)
        t = (target==label)
        inter = (p&t).sum().item()
        union = (p|t).sum().item()
        total = p.sum().item()+t.sum().item()
        # Skip absent classes rather than treating absence as perfect segmentation.
        if total == 0: continue
        dices.append(2*inter/total)
        ious.append(inter/union if union else 0.0)
    return {'mean_dice':float(np.mean(dices)),'mean_iou':float(np.mean(ious))}


def run_demo(epochs: int = 1, batch_size: int = 4, size: int = 32) -> dict[str,float]:
    seed_all()
    torch.set_num_threads(min(torch.get_num_threads(),2))
    data = SyntheticSegmentation(count=12,size=size)
    loader = DataLoader(data,batch_size=batch_size,shuffle=True)
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    model.train()
    last_loss = float('nan')
    for _ in range(epochs):
        for x,y in loader:
            optimizer.zero_grad()
            logits=model(x)
            loss=F.cross_entropy(logits,y)
            loss.backward()
            optimizer.step()
            last_loss=float(loss.item())
    model.eval()
    x,y=next(iter(DataLoader(data,batch_size=batch_size)))
    metrics=segmentation_metrics(model(x),y)
    return {'loss':last_loss,**metrics}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CPU-ready synthetic segmentation smoke test (not BraTS evaluation)')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--size',type=int,default=32)
    a=parser.parse_args()
    if a.epochs < 1: parser.error('--epochs must be >=1')
    print('SYNTHETIC DATA ONLY — demonstration metrics are not medical benchmarks')
    print(run_demo(epochs=a.epochs,size=a.size))
