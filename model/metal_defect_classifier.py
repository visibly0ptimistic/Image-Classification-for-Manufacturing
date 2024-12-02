import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from itertools import product
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def organize_dataset(root_dir):
    """
    Reorganize dataset to have proper train/test split for all classes
    """
    import tempfile
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    train_temp = os.path.join(temp_dir, 'train_temp')
    test_temp = os.path.join(temp_dir, 'test_temp')
    
    os.makedirs(train_temp, exist_ok=True)
    os.makedirs(test_temp, exist_ok=True)
    
    # Handle defect classes (from test directory)
    defect_classes = ['bent', 'color', 'scratch']
    for class_name in defect_classes:
        # Create class directories
        os.makedirs(os.path.join(train_temp, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_temp, class_name), exist_ok=True)
        
        # Get all images from test directory
        test_dir = os.path.join(root_dir, 'test', class_name)
        images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
        print(f"Found {len(images)} images for {class_name}")
        
        # Split 80% for training, 20% for testing
        num_train = int(len(images) * 0.8)
        train_images = images[:num_train]
        test_images = images[num_train:]
        
        # Copy files
        for img in train_images:
            shutil.copy2(
                os.path.join(test_dir, img),
                os.path.join(train_temp, class_name, img)
            )
        for img in test_images:
            shutil.copy2(
                os.path.join(test_dir, img),
                os.path.join(test_temp, class_name, img)
            )
    
    # Handle 'good' class (from train directory)
    os.makedirs(os.path.join(train_temp, 'good'), exist_ok=True)
    os.makedirs(os.path.join(test_temp, 'good'), exist_ok=True)
    
    # Get good images from train directory
    good_train_dir = os.path.join(root_dir, 'train', 'good')
    good_images = [f for f in os.listdir(good_train_dir) if f.endswith('.png')]
    print(f"Found {len(good_images)} good images")
    
    # Split good images
    num_train = int(len(good_images) * 0.8)
    train_images = good_images[:num_train]
    test_images = good_images[num_train:]
    
    # Copy good files
    for img in train_images:
        shutil.copy2(
            os.path.join(good_train_dir, img),
            os.path.join(train_temp, 'good', img)
        )
    for img in test_images:
        shutil.copy2(
            os.path.join(good_train_dir, img),
            os.path.join(test_temp, 'good', img)
        )
    
    # Replace original directories
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir, ignore_errors=True)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
    
    shutil.copytree(train_temp, train_dir)
    shutil.copytree(test_temp, test_dir)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    # Print final statistics
    print("\nFinal dataset organization:")
    for split in ['train', 'test']:
        print(f"\n{split.upper()} set:")
        for class_name in ['good'] + defect_classes:
            class_dir = os.path.join(root_dir, split, class_name)
            if os.path.exists(class_dir):
                n_images = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
                print(f"  {class_name}: {n_images} images")

class MetalDefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.classes = ['good', 'bent', 'color', 'scratch']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        # Load data
        data_dir = os.path.join(root_dir, mode)
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class DefectClassifier:
    def __init__(self, num_classes=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        
        # Initialize EfficientNetB0 with updated weights parameter
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )
        self.model = self.model.to(device)
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),  # Increased rotation
            transforms.RandomHorizontalFlip(),  # Add horizontal flip
            transforms.RandomVerticalFlip(),    # Add vertical flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Enhanced color jittering
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        return accuracy  # Only return accuracy for comparison

    def train_model(self, train_loader, val_loader, num_epochs=50,
                learning_rate=0.001, patience=5,
                optimizer=None):
        weights = torch.tensor([1.0, 10.0, 10.0, 10.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        best_val_acc = 0
        patience_counter = 0
        history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Calculate training metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = val_correct / val_total
            
            # Store metrics in history
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            
            scheduler.step()
            
            # Early stopping
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
            
            print(f'Epoch [{epoch+1}/{num_epochs}] '
                f'Train Acc: {100.*epoch_train_acc:.2f}% '
                f'Val Acc: {100.*epoch_val_acc:.2f}% '
                f'Train Loss: {epoch_train_loss:.4f} '
                f'Val Loss: {epoch_val_loss:.4f}')
        
        return best_val_acc, history

    def predict(self, loader):
        """Separate method for getting predictions and labels"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels)

def grid_search_cv(dataset, param_grid, k=3):
    """
    Perform grid search with k-fold cross-validation
    """
    learning_rates = param_grid['learning_rate']
    batch_sizes = param_grid['batch_size']
    optimizers = param_grid['optimizer']
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    best_params = None
    best_score = 0
    
    param_combinations = [
        {'lr': lr, 'batch_size': bs, 'optimizer': opt}
        for lr, bs, opt in product(learning_rates, batch_sizes, optimizers)
    ]
    
    for params in param_combinations:
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            classifier = DefectClassifier()
            
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                dataset, 
                batch_size=params['batch_size'],
                sampler=train_sampler
            )
            
            val_loader = DataLoader(
                dataset,
                batch_size=params['batch_size'],
                sampler=val_sampler
            )
            
            if params['optimizer'] == 'Adam':
                optimizer = optim.Adam(classifier.model.parameters(), lr=params['lr'])
            else:  # SGD with momentum
                optimizer = optim.SGD(classifier.model.parameters(), lr=params['lr'], momentum=0.9)
            
            val_score, _ = classifier.train_model(  # Unpack tuple return
                train_loader,
                val_loader,
                optimizer=optimizer,
                num_epochs=50  # Reduced for grid search
            )
            
            fold_scores.append(val_score)
        
        avg_score = np.mean(fold_scores)
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            
        print(f"Params: {params}, Average Score: {avg_score:.4f}")
    
    return best_params, best_score

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot confusion matrix using seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_training_history(train_metrics, val_metrics, metric_name):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        train_metrics (list): List of training metrics per epoch
        val_metrics (list): List of validation metrics per epoch
        metric_name (str): Name of the metric (e.g., 'Accuracy', 'Loss')
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Training {metric_name}')
    plt.plot(val_metrics, label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name} Over Time')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name.lower()}_history.png')
    plt.close()

def plot_class_distribution(dataset):
    """
    Plot the distribution of classes in the dataset.
    
    Args:
        dataset (MetalDefectDataset): The dataset to analyze
    """
    class_counts = {}
    for label in dataset.labels:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    plt.close()

def plot_sample_predictions(model, test_loader, device, classes):
    """
    Plot a grid of sample predictions with one example from each class from the test set.
    """
    model.eval()
    
    # Collect one example from each class
    test_examples = {cls: {'images': [], 'labels': []} for cls in classes}
    
    # Get test images and labels
    with torch.no_grad():
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                class_name = classes[lbl]
                if len(test_examples[class_name]['images']) == 0:  # Only take first example
                    test_examples[class_name]['images'].append(img)
                    test_examples[class_name]['labels'].append(lbl)
    
    # Prepare images for display
    display_images = []
    display_labels = []
    for cls in classes:
        if test_examples[cls]['images']:
            display_images.append(test_examples[cls]['images'][0])
            display_labels.append(test_examples[cls]['labels'][0])
    
    # Convert to tensors
    display_images = torch.stack(display_images).to(device)
    display_labels = torch.tensor(display_labels)
    
    # Get predictions
    outputs = model(display_images)
    predictions = outputs.max(1)[1].cpu()
    
    # Plot
    plt.figure(figsize=(20, 5))  # Wider figure for better spacing
    for i in range(len(classes)):
        plt.subplot(1, 4, i + 1)
        # Denormalize image properly
        img = display_images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        color = 'green' if predictions[i] == display_labels[i] else 'red'
        plt.title(f'True: {classes[display_labels[i]]}', 
                 color=color, pad=20)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', 
                bbox_inches='tight', 
                dpi=300,
                pad_inches=0.5)  # Add padding around the figure
    plt.close()

def plot_roc_curves(model, test_loader, device, classes):
    """
    Plot ROC curves for each class with different shapes and colors, horizontally offset.
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    plt.figure(figsize=(10, 8))
    
    # Define different markers and colors for each class
    markers = ['o', 's', '^', 'D']  # circle, square, triangle, diamond
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red
    offsets = [0.000, 0.001, 0.002, 0.003]  # Horizontal offsets
    
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Add small horizontal offset to separate overlapping curves
        fpr = np.clip(fpr + offsets[i], 0, 1)
        
        plt.plot(fpr, tpr, 
                color=colors[i],
                marker=markers[i],
                markevery=0.1,  # Show marker every 10% of points
                markersize=8,
                label=f'{classes[i]} (AUC = {roc_auc:.2f})',
                linewidth=2,
                markerfacecolor='white',  # White fill
                markeredgewidth=2)  # Thicker marker edge
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right", framealpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_all_visualizations(model, train_history, test_loader, dataset, device, classes):
    """
    Generate and save all visualizations.
    
    Args:
        model: The trained model
        train_history: Dictionary containing training metrics
        test_loader: DataLoader containing test samples
        dataset: The full dataset
        device: Device to run predictions on
        classes: List of class names
    """
    # Plot training history
    plot_training_history(train_history['train_acc'], train_history['val_acc'], 'Accuracy')
    if 'train_loss' in train_history:
        plot_training_history(train_history['train_loss'], train_history['val_loss'], 'Loss')
    
    # Plot class distribution
    plot_class_distribution(dataset)
    
    # Plot sample predictions
    plot_sample_predictions(model, test_loader, device, classes)
    
    # Plot ROC curves
    plot_roc_curves(model, test_loader, device, classes)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Dataset paths
    data_root = './metal_nut'
    
    # Reorganize dataset
    print("Reorganizing dataset...")
    organize_dataset(data_root)
    
    # Setup parameter grid (reduced for faster search)
    param_grid = {
        'learning_rate': [0.001],    # Just one best learning rate
        'batch_size': [32],          # Just one batch size
        'optimizer': ['Adam']        # Just Adam optimizer
    }
    
    # Create datasets
    full_dataset = MetalDefectDataset(data_root, transform=DefectClassifier().train_transform, mode='train')
    test_dataset = MetalDefectDataset(data_root, transform=DefectClassifier().test_transform, mode='test')
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(full_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Perform grid search
    print("Starting Grid Search...")
    best_params, best_score = grid_search_cv(full_dataset, param_grid)
    print("\nBest Parameters:", best_params)
    print(f"Best Cross-validation Score: {best_score:.4f}")
    
    # Train final model with best parameters
    print("\nTraining Final Model...")
    classifier = DefectClassifier()
    
    # Split dataset using stratified sampling
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_idx = next(sss.split(full_dataset.images, full_dataset.labels))

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])
    
    # Setup optimizer
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(classifier.model.parameters(), lr=best_params['lr'])
    else:
        optimizer = optim.SGD(classifier.model.parameters(), lr=best_params['lr'], momentum=0.9)
    
    # Train model and get history
    best_val_acc, history = classifier.train_model(train_loader, val_loader, optimizer=optimizer)
    print(f"Best validation accuracy: {100.*best_val_acc:.2f}%")
    
    # Load best model and evaluate
    classifier.model.load_state_dict(torch.load('best_model.pth'))
    test_acc = classifier.evaluate(test_loader)
    predictions, true_labels = classifier.predict(test_loader)
    
    # Generate all visualizations
    save_all_visualizations(
        model=classifier.model,
        train_history=history,
        test_loader=test_loader,
        dataset=full_dataset,
        device=classifier.device,
        classes=['good', 'bent', 'color', 'scratch']
    )
    
    # Print final results
    print("\nFinal Results:")
    print(f"Test Accuracy: {100.*test_acc:.2f}%")
    
    classes = ['good', 'bent', 'color', 'scratch']
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=classes, zero_division=1))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, classes)
    
    # Compare with benchmarks
    print("\nBenchmark Comparison:")
    print(f"Current Model Accuracy: {100.*test_acc:.2f}%")
    print("SVM with HOG Benchmark: 85.00%")
    print("ResNet50 Benchmark (Wang et al., 2017): 91.00%")
    print("EfficientNetB0 Benchmark (Chen et al., 2022): 93.00%")

if __name__ == '__main__':
    main()