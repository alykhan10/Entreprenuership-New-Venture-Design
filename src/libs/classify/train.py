import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preprocessing import load_dataset, split_dataset, create_data_loaders
from model import create_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data_dir = 'C:/Users/connor/Pictures/Camera Roll/dataset'
dataset = load_dataset(data_dir)
train_dataset, val_dataset, test_dataset = split_dataset(dataset)
train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

# Create model and move it to the GPU
model = create_model(num_classes=4, model_name="resnet50").to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print training loss
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Print validation accuracy
    print(f"Validation Accuracy: {100 * correct / total}%")

# Save the model
torch.save(model.state_dict(), 'tool_classifier_new.pth')