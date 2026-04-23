import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

from model import SelfPruningNet
from config import (
    SEED,
    BATCH_SIZE,
    EPOCHS,
    LR,
    DEVICE,
    LAMBDAS,
    OUTPUT_DIR,
    CHECKPOINT_DIR
)
from utils import (
    seed_everything,
    create_folders,
    save_plot,
    save_gate_histogram
)
from evaluate import (
    evaluate,
    get_sparsity,
    collect_gate_values
)


def get_dataloaders():
    """
    CIFAR-10 loaders
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return trainloader, testloader


def train_model(lambda_sparse, trainloader, testloader):
    """
    Train one model for one lambda value.
    """
    model = SelfPruningNet().to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        model.train()

        temperature = max(0.7, 4.0 * (0.93 ** epoch))

        loop = tqdm(trainloader)

        for x, y in loop:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(x, temperature)

            ce_loss = criterion(outputs, y)
            sparse_loss = model.sparsity_loss(temperature)

            loss = ce_loss + lambda_sparse * sparse_loss

            loss.backward()
            optimizer.step()

            loop.set_description(
                f"Lambda {lambda_sparse} | Epoch {epoch+1}"
            )

            loop.set_postfix(
                loss=round(loss.item(), 4)
            )

        scheduler.step()

        acc = evaluate(model, testloader)
        sparsity = get_sparsity(model)

        print(
            f"Epoch {epoch+1}: "
            f"Accuracy={acc:.2f}% | "
            f"Sparsity={sparsity:.2f}%"
        )

    return model


def main():

    seed_everything(SEED)

    create_folders(
        OUTPUT_DIR,
        CHECKPOINT_DIR
    )

    trainloader, testloader = get_dataloaders()

    results = []

    best_model = None
    best_accuracy = 0

    for lam in LAMBDAS:

        print("=" * 60)
        print("Training with lambda =", lam)

        model = train_model(
            lam,
            trainloader,
            testloader
        )

        acc = evaluate(model, testloader)
        sparsity = get_sparsity(model)

        torch.save(
            model.state_dict(),
            f"{CHECKPOINT_DIR}/model_lambda_{lam}.pth"
        )

        results.append([
            lam,
            acc,
            sparsity
        ])

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model

    df = pd.DataFrame(
        results,
        columns=[
            "Lambda",
            "Test Accuracy",
            "Sparsity %"
        ]
    )

    df.to_csv(
        f"{OUTPUT_DIR}/results.csv",
        index=False
    )

    print("\nFinal Results:")
    print(df)

    save_plot(df)

    gates = collect_gate_values(best_model)
    save_gate_histogram(gates)

    print("\nArtifacts saved successfully.")


if __name__ == "__main__":
    main()