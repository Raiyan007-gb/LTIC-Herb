
import torch
import torch.nn as nn
from PIL import Image
import open_clip
from torch.cuda.amp import autocast  # For mixed precision training

class CLIP_VIT_Large(nn.Module):
    def __init__(self, num_classes=1000, dropout=False, gamma=0.5):
        super(CLIP_VIT_Large, self).__init__()
        
        # Move model to GPU immediately
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = outputs[0].to(self.device)
        self.clip_preprocess = outputs[1]

        # Enable parallel processing if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            self.clip_model = nn.DataParallel(self.clip_model)

        self.gamma = gamma
        
        # Increase network capacity with larger layers
        self.intermediate = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 170)
        ).to(self.device)
        
        self.fc = nn.Linear(56, num_classes).to(self.device)
        self.dropout_mark = dropout
        self.dropout = nn.Dropout(p=0.5) if dropout else None

        # Enable cudnn benchmarking for faster training
        torch.backends.cudnn.benchmark = True

    @autocast()  # Enable automatic mixed precision
    def _forward_impl(self, x):
        batch_size = x.size(0)
        
        # Pre-allocate GPU memory for efficiency
        with torch.cuda.amp.autocast():  # Use mixed precision
            with torch.no_grad():
                x = self.clip_model.encode_image(x)
            
            x = torch.flatten(x, 1)
            
            # Process in parallel using larger batch sizes
            x = self.intermediate(x)
            
            # Split features
            c = x.size(1) // 3
            x1, x2, x3 = x[:, :c], x[:, c:c*2], x[:, c*2:c*3]
            
            # Concatenate efficiently
            out = torch.cat((x1, x2, x3), dim=0)
            
            if self.dropout_mark:
                out = self.dropout(out)
            
            if self.training:
                y = self.fc(out)
            else:
                weight = self.fc.weight
                norm = torch.norm(weight, 2, 1, keepdim=True)
                weight = weight / torch.pow(norm, self.gamma)
                y = torch.mm(out, torch.t(weight))
            
            return y[:batch_size], y[batch_size:batch_size*2], y[batch_size*2:batch_size*3]

    def forward(self, x):
        return self._forward_impl(x)