import torch
from data_preprocessing import load_dataset, split_dataset, create_data_loaders
from model import create_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
data_dir = 'path_to_dataset'
dataset = load_dataset(data_dir)
_, _, test_dataset = split_dataset(dataset)
_, _, test_loader = create_data_loaders(None, None, test_dataset)

# Load the trained model and move it to the GPU
model = create_model(num_classes=4).to(device)
model.load_state_dict(torch.load('tool_classifier.pth'))
model.eval()

# Evaluate the model
test_loss = 0.0
correct = 0
total = 0
criterion = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")