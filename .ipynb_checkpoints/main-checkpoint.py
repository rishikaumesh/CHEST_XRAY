
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd

from dataset import PneumoniaDataset, train_transform, val_transform
from model import MyModel
from trainer import train_model
from args import get_args
from evaluate import evaluate_model


def main():
    """
    Main function to train and evaluate the model using cross-validation.
    """
    # Parse arguments
    args = get_args()

    # Print configuration settings
    print(f"Using dropout rate: {args.dropout_rate}")
    print(f"Using L2 regularization: {args.l2_reg}")
    print(f"Early stopping enabled: {args.early_stopping} with patience {args.patience}")
    print(f"Using weighted loss: {args.use_weighted_loss}")

    best_model_path = os.path.join(args.out_dir, "best_model_overall.pth")
    best_overall_accuracy = 0.0

    # Cross-validation loop
    for fold in range(5):
        print(f"\n========== Fold {fold} ==========")

        # Load training and validation data
        train_set = pd.read_csv(os.path.join(args.csv_dir, f"fold_{fold}_train.csv"))
        val_set = pd.read_csv(os.path.join(args.csv_dir, f"fold_{fold}_val.csv"))

        # Preparing datasets
        train_dataset = PneumoniaDataset(dataset_df=train_set, transform=train_transform)
        val_dataset = PneumoniaDataset(dataset_df=val_set, transform=val_transform)

        # Create data loaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=torch.cuda.is_available()
        )

        # Initialize the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyModel(backbone=args.backbone, num_classes=2, dropout_rate=args.dropout_rate).to(device)

        # Train the model
        fold_model_path, fold_accuracy = train_model(model, train_loader, val_loader, args, fold)

        # Keep track of the best model across folds
        if fold_accuracy > best_overall_accuracy:
            best_overall_accuracy = fold_accuracy
            torch.save(model.state_dict(), best_model_path)  # Save the best overall model
            print(f"New best overall model found on Fold {fold} with Accuracy: {fold_accuracy:.4f}")

    # Evaluate the best model on the test dataset
    print("\nEvaluating the best model on the test dataset...")
    test_data = pd.read_csv(args.test_path)
    test_dataset = PneumoniaDataset(dataset_df=test_data, transform=val_transform, test_mode=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
    )

    # Load the best model and evaluate on test data
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()

