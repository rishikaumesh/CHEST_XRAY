import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Define paths
train_path = '/home/user/persistent/chest_xray/train'  # Original train folder
val_path = '/home/user/persistent/chest_xray/val'  # Original val folder
combined_path = '/home/user/persistent/chest_xray/train_val'  # Path for combined train+val folder
test_path = '/home/user/persistent/chest_xray/test'  # Test folder remains as is
out_dir = '/home/user/persistent/chest_xray/data/CSVs'  # Directory to save the CSVs

# Ensure the combined train_val directory exists
os.makedirs(combined_path, exist_ok=True)
for class_name in ['NORMAL', 'PNEUMONIA']:
    os.makedirs(os.path.join(combined_path, class_name), exist_ok=True)

# Combine train and val datasets into train_val
for class_name in ['NORMAL', 'PNEUMONIA']:
    for folder_path in [train_path, val_path]:
        source_folder = os.path.join(folder_path, class_name)
        target_folder = os.path.join(combined_path, class_name)
        for image_name in os.listdir(source_folder):
            source_image_path = os.path.join(source_folder, image_name)
            target_image_path = os.path.join(target_folder, image_name)
            if not os.path.exists(target_image_path):  # Avoid overwriting
                shutil.copy(source_image_path, target_image_path)

print(f"Train and val datasets merged into {combined_path}")

# Process the combined train_val dataset
data = []
classes = ['NORMAL', 'PNEUMONIA']
for label, class_name in enumerate(classes):
    folder_path = os.path.join(combined_path, class_name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        name = os.path.splitext(image_name)[0]
        data.append({'Path': image_path, 'Label': label, 'Name': name})

df = pd.DataFrame(data)

# Display sample data
print("Sample data from the merged dataset:")
print(df.head(10))

# Perform Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate CSV files for each fold
for fold, (train_index, val_index) in enumerate(skf.split(df, df['Label'])):
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]
    
    train_csv_path = os.path.join(out_dir, f'fold_{fold}_train.csv')
    val_csv_path = os.path.join(out_dir, f'fold_{fold}_val.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Saved fold {fold} train and validation CSVs.")

# Prepare the test set
test_data = []
for label, class_name in enumerate(classes):
    folder_path = os.path.join(test_path, class_name)
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        name = os.path.splitext(image_name)[0]
        test_data.append({'Path': image_path, 'Label': label, 'Name': name})

test_df = pd.DataFrame(test_data)

test_csv_path = os.path.join(out_dir, 'test.csv')
test_df.to_csv(test_csv_path, index=False)

print("Test set saved to CSV.")
print("All folds have been processed and saved.")
