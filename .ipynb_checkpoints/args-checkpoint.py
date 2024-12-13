import argparse
import os

def get_args():
    """
    Defines and parses command-line arguments for the pneumonia detection project.
    """
    parser = argparse.ArgumentParser(description='Pneumonia Detection')

    # Paths for the dataset and outputs
    base_dir = '/home/user/persistent/chest_xray'
    parser.add_argument('--train_val_path', type=str, default=os.path.join(base_dir, 'train_val'),
                        help='Path to the combined train and validation dataset.')
    parser.add_argument('--test_path', type=str, default=os.path.join(base_dir, 'data/CSVs', 'test.csv'),
                        help='Path to the test dataset CSV file.')
    parser.add_argument('--csv_dir', type=str, default=os.path.join(base_dir, 'data/CSVs'),
                        help='Directory to save the generated CSV files.')
    parser.add_argument('--out_dir', type=str, default=os.path.join(base_dir, 'session'),
                        help='Directory to save outputs like models, plots, etc.')

    # Model arguments
    parser.add_argument('--model_path', type=str, default=os.path.join(base_dir, 'session', 'best_model_overall.pth'),
                        help='Path to the best model checkpoint for evaluation.')
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'custom'],
                        default='resnet18', help='Model backbone to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for regularization.')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, choices=[16, 32, 64],
                        help='Batch size for training and validation.')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--fold', type=int, default=0, help='Current fold number for cross-validation.')

    # Advanced training options
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on validation performance.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--use_weighted_loss', action='store_true',
                        help='Use weighted loss function to address class imbalance.')
    parser.add_argument('--l2_reg', type=float, default=0.0, help='L2 regularization weight.')

    # Data augmentation
    parser.add_argument('--use_data_augmentation', action='store_true',
                        help='Enable data augmentation for training.')

    # Optimizer arguments
    parser.add_argument('--b1', type=float, default=0.9,
                        help='Decay of first-order momentum of gradient for Adam.')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='Decay of second-order momentum of gradient for Adam.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print("Parsed Arguments:")
    print(args)

