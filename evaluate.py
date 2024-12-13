# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score
# import os
# import pandas as pd
# from dataset import PneumoniaDataset, val_transform
# from model import MyModel
# from args import get_args


# def evaluate_model(model, data_loader, device):
#     """
#     Evaluate the model on the provided dataset.
#     """
#     model.eval()
#     all_targets = []
#     all_predictions = []
#     all_probs = []
#     running_loss = 0.0
#     criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class logits

#     with torch.no_grad():
#         for batch in data_loader:
#             inputs = batch['img'].to(device)
#             targets = batch['target'].long().to(device)

#             outputs = model(inputs)  # Outputs logits for both classes
#             probs = torch.softmax(outputs, dim=1)  # Compute probabilities
#             _, predicted = torch.max(probs, 1)  # Predicted class

#             loss = criterion(outputs, targets)  # Loss for multi-class
#             running_loss += loss.item()

#             all_targets.extend(targets.cpu().numpy())
#             all_predictions.extend(predicted.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())

#     avg_loss = running_loss / len(data_loader)
#     roc_auc = roc_auc_score(all_targets, [p[1] for p in all_probs])  # Use probabilities for class 1
#     avg_precision = average_precision_score(all_targets, [p[1] for p in all_probs])
#     balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)

#     print(f'Loss: {avg_loss:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}, '
#           f'ROC-AUC: {roc_auc:.4f}, Average Precision: {avg_precision:.4f}')

#     return avg_loss, balanced_accuracy, roc_auc, avg_precision


# def main():
#     """
#     Main function to evaluate the best model on the test dataset.
#     """
#     args = get_args()  # Parse arguments
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the test dataset
#     print(f"Loading test dataset from: {args.test_path}")
#     test_data = pd.read_csv(args.test_path)
#     test_dataset = PneumoniaDataset(dataset_df=test_data, transform=val_transform, test_mode=True)
#     test_loader = DataLoader(
#         dataset=test_dataset,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=torch.cuda.is_available()
#     )

#     # Initialize the model
#     model = MyModel(backbone=args.backbone, num_classes=2, dropout_rate=args.dropout_rate)

#     # Load the best model's state
#     if not os.path.exists(args.model_path):
#         raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

#     print(f"Loading model checkpoint from: {args.model_path}")
#     model.load_state_dict(torch.load(args.model_path))
#     model.to(device)

#     # Evaluate the model on the test dataset
#     print("Evaluating the best model on the test dataset...")
#     avg_loss, balanced_accuracy, roc_auc, avg_precision = evaluate_model(model, test_loader, device)

#     # Save evaluation results to a text file
#     results_path = os.path.join(args.out_dir, "test_evaluation_results.txt")
#     with open(results_path, "w") as f:
#         f.write(f"Test Loss: {avg_loss:.4f}\n")
#         f.write(f"Balanced Accuracy: {balanced_accuracy:.4f}\n")
#         f.write(f"ROC-AUC: {roc_auc:.4f}\n")
#         f.write(f"Average Precision: {avg_precision:.4f}\n")
#     print(f"Test evaluation results saved to: {results_path}")


# if __name__ == "__main__":
#     main()


import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataset import PneumoniaDataset, val_transform
from model import MyModel
from args import get_args
from torch.utils.data import DataLoader


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the provided dataset and visualize results.
    """
    model.eval()
    all_targets = []
    all_predictions = []
    all_probs = []
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
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

    avg_loss = running_loss / len(data_loader)
    roc_auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
    avg_precision = average_precision_score(all_targets, [p[1] for p in all_probs])
    balanced_accuracy = balanced_accuracy_score(all_targets, all_predictions)

    print(f'Loss: {avg_loss:.4f}, Balanced Accuracy: {balanced_accuracy:.4f}, '
          f'ROC-AUC: {roc_auc:.4f}, Average Precision: {avg_precision:.4f}')

    # Visualizations
    visualize_evaluation(all_targets, all_predictions, all_probs, roc_auc, avg_precision)

    return avg_loss, balanced_accuracy, roc_auc, avg_precision


def visualize_evaluation(all_targets, all_predictions, all_probs, roc_auc, avg_precision):
    """
    Generate and save visualizations for the evaluation results.
    """
    # 1. Confusion Matrix
    cm = confusion_matrix(all_targets, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(all_targets, [p[1] for p in all_probs])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_display.plot()
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png")
    plt.show()

    # 3. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(all_targets, [p[1] for p in all_probs])
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision)
    pr_display.plot()
    plt.title("Precision-Recall Curve")
    plt.savefig("precision_recall_curve.png")
    plt.show()


def main():
    """
    Main function to evaluate the best model on the test dataset.
    """
    args = get_args()  # Parse arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    print(f"Loading test dataset from: {args.test_path}")
    test_data = pd.read_csv(args.test_path)
    test_dataset = PneumoniaDataset(dataset_df=test_data, transform=val_transform, test_mode=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available()
    )

    # Initialize the model
    model = MyModel(backbone=args.backbone, num_classes=2, dropout_rate=args.dropout_rate)

    # Load the best model's state
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    print(f"Loading model checkpoint from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Evaluate the model on the test dataset
    print("Evaluating the best model on the test dataset...")
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()

