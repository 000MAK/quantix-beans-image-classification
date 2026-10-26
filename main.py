"""Train a compact CNN on the online Beans leaf image dataset."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def validate_epochs(epochs: int) -> int:
    if int(epochs) < 1: raise ValueError("epochs must be at least 1")
    return int(epochs)

def main(output_dir="outputs", epochs=3, batch_size=32, image_size=96, seed=42):
    validate_epochs(epochs)
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from datasets import load_dataset
    import numpy as np
    torch.manual_seed(seed)
    ds=load_dataset("beans")
    train_rows=ds["train"]
    eval_rows=ds["validation"] if "validation" in ds else ds["test"]
    transform=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])
    class HubImages(Dataset):
        def __init__(self, rows): self.rows=rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, i):
            row=self.rows[i]; image=row["image"].convert("RGB")
            return transform(image), int(row["labels"] if "labels" in row else row["label"])
    train_loader=DataLoader(HubImages(train_rows),batch_size=batch_size,shuffle=True,num_workers=0)
    eval_loader=DataLoader(HubImages(eval_rows),batch_size=batch_size,shuffle=False,num_workers=0)
    class SmallCNN(nn.Module):
        def __init__(self, classes):
            super().__init__(); self.features=nn.Sequential(
              nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
              nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
              nn.Conv2d(64,96,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)))
            self.head=nn.Linear(96,classes)
        def forward(self,x): return self.head(self.features(x).flatten(1))
    label_col="labels" if "labels" in train_rows.column_names else "label"
    n_classes=len(set(train_rows[label_col])); model=SmallCNN(n_classes)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=nn.CrossEntropyLoss()
    for epoch in range(validate_epochs(epochs)):
        model.train(); total=0.0
        for x,y in train_loader:
            optimizer.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); optimizer.step(); total += loss.item()*len(y)
        print(f"epoch={epoch+1} train_loss={total/max(len(train_rows),1):.4f}")
    model.eval(); correct=seen=0
    with torch.no_grad():
        for x,y in eval_loader:
            pred=model(x).argmax(1); correct += int((pred==y).sum()); seen += len(y)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/"beans_cnn.pt")
    report={"dataset":"Hugging Face beans","train_rows":len(train_rows),"evaluation_rows":seen,
      "classes":n_classes,"accuracy":correct/max(seen,1),"epochs":epochs,"seed":seed,
      "warning":"Small educational baseline. Review dataset terms, split design and class-wise results before making performance claims."}
    (out/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",default="outputs"); p.add_argument("--epochs",type=int,default=3); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--image-size",type=int,default=96); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); print(json.dumps(main(a.output_dir,a.epochs,a.batch_size,a.image_size,a.seed),indent=2))
