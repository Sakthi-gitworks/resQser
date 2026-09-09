import torch
import torch.nn as nn

device = torch.device("cpu")

# Exact architecture matching custom_accident_model.pth
class CustomAccidentModel(nn.Module):
    def __init__(self):
        super(CustomAccidentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # features.0
            nn.ReLU(),                                    # features.1
            nn.MaxPool2d(2, 2),                           # features.2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # features.3
            nn.ReLU(),                                    # features.4
            nn.MaxPool2d(2, 2),                           # features.5
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # features.6
            nn.ReLU(),                                    # features.7
            nn.MaxPool2d(2, 2)                            # features.8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # classifier.0
            nn.Linear(100352, 256),                       # classifier.1
            nn.ReLU(),                                    # classifier.2
            nn.Dropout(0.5),                              # classifier.3
            nn.Linear(256, 2)                             # classifier.4
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 1. Instantiate the matching custom architecture
model = CustomAccidentModel().to(device)

# 2. Load trained weights
model.load_state_dict(torch.load("custom_accident_model.pth", map_location=device))
model.eval()

# 3. Create dummy input (1 image, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# 4. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✅ Successfully converted custom_accident_model.pth -> model.onnx")