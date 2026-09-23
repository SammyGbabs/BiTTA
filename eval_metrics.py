
import argparse, torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
from models.EfficientNet import EfficientNetB0Dropout

CLASSES = ["cbb", "cbsd", "cgm", "cmd", "healthy"]
HEALTHY_IDX = CLASSES.index("healthy")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def evaluate(checkpoint_path, dataset_root, label=""):
    net = EfficientNetB0Dropout(num_classes=5)
    ckpt = torch.load(checkpoint_path, map_location="cuda")
    net.load_state_dict(ckpt)
    net = net.cuda().eval()

    ds = datasets.ImageFolder(dataset_root, transform=TRANSFORM)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = net(x.cuda(), dropout=0.0)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    disease_mask = all_labels != HEALTHY_IDX
    disease_to_healthy = (all_preds[disease_mask] == HEALTHY_IDX).mean()

    print(f"\n=== {label or checkpoint_path} on {dataset_root} ===")
    print(f"Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Disease->healthy rate: {disease_to_healthy:.4f}")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=3))
    return {"accuracy": acc, "macro_f1": macro_f1, "disease_to_healthy": disease_to_healthy, "confusion_matrix": cm.tolist()}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()
    evaluate(args.checkpoint, args.dataset_root, args.label)
