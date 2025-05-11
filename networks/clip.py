import torch
import torch.nn as nn
from PIL import Image
import open_clip

class CLIP_VIT(nn.Module):
    def __init__(self, num_classes=1000, dropout=False, gamma=0.5):
        super(CLIP_VIT, self).__init__()
        # Load CLIP model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Correctly handle the output of create_model_and_transforms
        outputs = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = outputs[0]  # The CLIP model itself
        self.clip_preprocess = outputs[1]  # Preprocessing transforms

        self.gamma = gamma
        
        # Fully connected layer for classification
        self.fc = nn.Linear(170, num_classes)  # Adjust input dimensions if needed
        self.dropout_mark = dropout
        self.dropout = nn.Dropout(p=0.5) if dropout else None

    def _forward_impl(self, x):
        """
        Forward implementation using CLIP image encoder.
        """
        # Use CLIP encoder for feature extraction
        with torch.no_grad():
            x = self.clip_model.encode_image(x)  # CLIP encoding
            # print(f"Features after CLIP encoder: {x.shape}")  # Debugging dimensions

        # Flatten the CLIP output
        x = torch.flatten(x, 1)
        # print(f"Features after flattening: {x.shape}")

        # Divide into three parts
        c = x.size(1) // 3  # Ensure the feature dimensions are divisible by 3
        bt = x.size(0)
        x1, x2, x3 = x[:, :c], x[:, c:c*2], x[:, c*2:c*3]

        # Concatenate outputs
        out = torch.cat((x1, x2, x3), dim=0)

        # Apply dropout if enabled
        if self.dropout_mark:
            out = self.dropout(out)

        # Training logic
        if self.training:
            y = self.fc(out)
        else:
            # Normalize weights for testing
            weight = self.fc.weight
            norm = torch.norm(weight, 2, 1, keepdim=True)
            weight = weight / torch.pow(norm, self.gamma)
            y = torch.mm(out, torch.t(weight))

        # Return the three splits
        return y[:bt, :], y[bt:bt*2, :], y[bt*2:bt*3, :]

    def forward(self, x):
        return self._forward_impl(x)

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = ResNet_ResLT_with_CLIP(num_classes=10, dropout=True, gamma=0.5)

    # Switch to training mode
    model.train()

    # Example input tensor (batch of images with shape [batch_size, channels, height, width])
    input_tensor = torch.randn(8, 3, 224, 224)

    # Perform forward pass
    output = model(input_tensor)

    # Output dimensions
    print(f"Output dimensions: {[o.shape for o in output]}")
