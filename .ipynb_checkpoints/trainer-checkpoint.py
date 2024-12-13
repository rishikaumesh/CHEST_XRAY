import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from dataset import PneumoniaDataset, train_transform, val_transform

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, args, fold):
    """
    Trains the model for a specific fold and tracks the best validation performance.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        args (Namespace): Parsed arguments from args.py.
        fold (int): Current fold number.

    Returns:
        str: Path to the best model for this fold.
        float: Best validation accuracy for this fold.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    val_balanced_accuracies, val_roc_aucs, val_avg_precisions = [], [], []

    best_val_accuracy = 0.0
    best_model_path = None

    for epoch in range(args.epochs):
        running_loss = 0.0

        # Training phase
        model.train()

        for batch in train_loader:
            inputs = batch['img'].to(device)
            targets = batch['target'].long().to(device)  # Convert targets to torch.int64
        
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # CrossEntropyLoss requires int64 targets
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()


        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Fold {fold} | Epoch {epoch + 1}/{args.epochs} - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        val_loss, val_accuracy, roc_auc, avg_precision = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_balanced_accuracies.append(val_accuracy)
        val_roc_aucs.append(roc_auc)
        val_avg_precisions.append(avg_precision)

        print(f"Fold {fold} | Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
              f"ROC-AUC: {roc_auc:.4f}, Avg Precision: {avg_precision:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(args.out_dir, f'best_model_fold_{fold}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved for Fold {fold} at epoch {epoch + 1} with accuracy {val_accuracy:.4f}")

    # Plot metrics for this fold
    plot_metrics(train_losses, val_losses, val_balanced_accuracies, val_roc_aucs, val_avg_precisions, args.out_dir, fold)

    return best_model_path, best_val_accuracy


def validate_model(model, val_loader, criterion):
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: Validation loss, balanced accuracy, ROC-AUC, average precision.
    """
    model.eval()
    running_loss = 0.0
    all_targets, all_predictions, all_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)
            targets = batch['target'].long().to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)
    roc_auc = roc_auc_score(all_targets, [p[1] for p in all_probs])  # Fix for binary classification
    avg_precision = average_precision_score(all_targets, [p[1] for p in all_probs])

    return avg_loss, balanced_accuracy, roc_auc, avg_precision



def plot_metrics(train_losses, val_losses, val_accuracies, val_roc_aucs, val_precisions, output_dir, fold):
    """
    Plots metrics for training and validation.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        val_accuracies (list): List of validation accuracies.
        val_roc_aucs (list): List of validation ROC-AUCs.
        val_precisions (list): List of validation average precisions.
        output_dir (str): Directory to save the plots.
        fold (int): Current fold number.
    """
    epochs = range(len(train_losses))

    # Plot losses
    plt.figure()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title(f"Loss Over Epochs - Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"loss_plot_fold_{fold}.png"))

    # Plot accuracies
    plt.figure()
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title(f"Balanced Accuracy Over Epochs - Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"accuracy_plot_fold_{fold}.png"))

    # Plot ROC-AUC
    plt.figure()
    plt.plot(epochs, val_roc_aucs, label="Validation ROC-AUC")
    plt.title(f"ROC-AUC Over Epochs - Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"roc_auc_plot_fold_{fold}.png"))

    # Plot average precision
    plt.figure()
    plt.plot(epochs, val_precisions, label="Validation Average Precision")
    plt.title(f"Average Precision Over Epochs - Fold {fold}")
    plt.xlabel("Epochs")
    plt.ylabel("Average Precision")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"avg_precision_plot_fold_{fold}.png"))

    print(f"Plots saved for Fold {fold}.")
