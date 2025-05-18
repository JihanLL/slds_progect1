import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from models.cnn import CNNv4
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from utils import extract_features


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "22355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, num_classes):
    setup(rank, world_size)

    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root="./data", train=True, download=False, transform=transform
    )
    testset = datasets.MNIST(root="./data", train=False, transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    model = CNNv4(num_classes=num_classes).to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc, test_acc = [], []

    for epoch in range(10):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Clip target to available classes if too many
            target = torch.clamp(target, max=num_classes - 1)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        train_acc.append(correct / total)

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = torch.clamp(target, max=num_classes - 1)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        acc = correct / total
        test_acc.append(acc)

        if rank == 0:
            print(f"[{epoch + 1}] Train Acc: {train_acc[-1]:.4f}, Test Acc: {acc:.4f}")

    if rank == 0:
        plt.plot(train_acc, label="Train Acc")
        plt.plot(test_acc, label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Class={num_classes}")
        plt.legend()
        plt.savefig(f"results/class_{num_classes}.png")
        print(f"Saved plot to results/class_{num_classes}.png")

    if rank == 0:
        feats, lbls = extract_features(model, test_loader, device, num_classes)

        for method in ["pca", "tsne"]:
            if method == "pca":
                reducer = PCA(n_components=2)
            else:
                reducer = TSNE(n_components=2, init="pca", random_state=42)

            reduced = reducer.fit_transform(feats.numpy())
            plt.figure(figsize=(8, 6))
            for i in range(min(num_classes, 10)):
                idx = (lbls == i).numpy()
                plt.scatter(
                    reduced[idx, 0], reduced[idx, 1], label=f"Class {i}", alpha=0.6
                )
            plt.legend()
            plt.title(
                f"Feature distribution ({method.upper()}) - Class={num_classes}"
            )
            os.makedirs("results", exist_ok=True)
            plt.savefig(f"results/features_{method}_class_{num_classes}.png")
            print(
                f"[{method.upper()}] Feature plot saved to results/features_{method}_class_{num_classes}.png"
            )

    cleanup()


def main():
    world_size = torch.cuda.device_count()
    for num_classes in [10, 12, 14, 16, 18, 20]:
        print(f"\nLaunching training with {num_classes} classes...")
        mp.spawn(train, args=(world_size, num_classes), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
