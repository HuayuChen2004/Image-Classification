import torch
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
import time
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preds_and_probs(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)             
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


class BaseModule(nn.Module):
    def train_model(self, train_loader, criterion, optimizer, num_epochs=10):
        self.to(device)
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (pbar.n+1))
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")
        print("Finished Training")
        return self

    def evaluate_model(self, loader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        acc = 100 * correct / total
        print(f"Accuracy: {acc:.2f}%")
        return acc

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
        print(f"Model loaded from {path}")
        return self

class CNN(BaseModule):
    def __init__(self, num_classes=10, image_size=(384, 384)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * (image_size[0] // 8) * (image_size[1] // 8), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.convnet = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            self.conv3,
            nn.ReLU(),
            self.pool
        )
        self.fcnet = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fcnet(x)
        return x


class CNN_structure_improve(BaseModule):
    def __init__(self, num_classes=10, image_size=(384, 384), dropout_conv=0.2, dropout_fc=0.5):
        super().__init__()
        C, H, W = 3, image_size[0], image_size[1]
        self.block1 = nn.Sequential(
            nn.Conv2d(C,  32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_conv)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_conv)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_conv)
        )
        flattened_size = 128 * (H // 8) * (W // 8)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)              
        x = self.block2(x)              
        x = self.block3(x)              
        x = x.view(x.size(0), -1)       
        x = self.classifier(x)          
        return x

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model with K-Fold Cross-Validation")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for K-Fold Cross-Validation")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the dataset")
    parser.add_argument("--image_size", nargs=2, type=int, default=(384, 384), help="Size of input images, e.g. --image_size 384 384")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output files")
    parser.add_argument("--save_model", action="store_true", help="Flag to save the trained model")
    parser.add_argument("--save_metrics", action="store_true", help="Flag to save the metrics")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--augment_data", action="store_true", help="Flag to apply data augmentation")
    parser.add_argument("--name", type=str, default="baseline", help="Name of the model")
    parser.add_argument("--structure_improve", action="store_true", help="Flag to use improved structure")
    parser.add_argument("--dropout_conv", type=float, default=0.2, help="Dropout rate for convolutional layers")
    parser.add_argument("--dropout_fc", type=float, default=0.5, help="Dropout rate for fully connected layers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if not args.augment_data:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=384, scale=(0.8, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    full_dataset = datasets.ImageFolder("garbage-dataset-resized", transform=transform)
    class_names = full_dataset.classes
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)

    k = args.num_folds
    kf = KFold(n_splits=k, shuffle=True, random_state=args.seed)
    fold_accuracies = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n=== Fold {fold}/{k} ===")

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler   = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(
            full_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader   = DataLoader(
            full_dataset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=True
        )
        if args.structure_improve:
            model = CNN_structure_improve(num_classes=args.num_classes, image_size=args.image_size, dropout_conv=args.dropout_conv, dropout_fc=args.dropout_fc).to(device)
        else:
            model = CNN(num_classes=args.num_classes, image_size=args.image_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        model.train_model(train_loader, criterion, optimizer, num_epochs=args.num_epochs)

        y_true, y_pred, y_prob = get_preds_and_probs(model, val_loader, device)

        acc = accuracy_score(y_true, y_pred)

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=range(args.num_classes), zero_division=0
        )
        f1_macro = np.mean(f1)

        cm = confusion_matrix(y_true, y_pred, labels=range(args.num_classes))

        y_true_bin = label_binarize(y_true, classes=range(args.num_classes))
        roc_auc = roc_auc_score(y_true_bin, y_prob, average="macro")

        fold_metrics.append({
            "accuracy":    acc,
            "f1_macro":    f1_macro,
            "roc_auc":     roc_auc,
            "precision":   prec, 
            "recall":      rec,   
            "f1_per_cls":  f1,    
            "support":     sup,   
            "confusion":   cm,     
        })

        print(f" Fold {fold} — Acc: {acc:.4f}, F1‑macro: {f1_macro:.4f}, ROC‑AUC: {roc_auc:.4f}")

    mean_acc = sum(fold_accuracies) / k
    print(f"\nAverage validation accuracy over {k} folds: {mean_acc:.2f}%")

    if args.save_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        save_dir = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args.output_dir, save_dir)
        os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        print(f"Metrics saved to {os.path.join(output_dir, 'metrics.csv')}")

    if args.save_model:

        with open(os.path.join(output_dir, "name.txt"), "w") as f:
            f.write(args.name)

        class_names_df = pd.DataFrame(class_names, columns=["class_name"])
        class_names_df.to_csv(os.path.join(output_dir, "class_names.csv"), index=False)
        print(f"Class names saved to {os.path.join(output_dir, 'class_names.csv')}")

        args_dict = vars(args)
        args_df = pd.DataFrame([args_dict])
        args_df.to_csv(os.path.join(output_dir, "args.csv"), index=False)
        print(f"Arguments saved to {os.path.join(output_dir, 'args.csv')}")

        model_path = os.path.join(output_dir, "model.pth")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")



