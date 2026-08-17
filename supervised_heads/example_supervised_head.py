import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleSupervisedHead(nn.Module):
    """
    Example supervised head for testing integration with LeJEPA/DINO frameworks.
    
    This head supports multiple task types:
    - classification: Single-label classification
    - multilabel: Multi-label classification with sigmoid outputs
    - segmentation: Pixel-wise segmentation
    
    Args:
        config: Configuration object with auxiliary parameters
        input_dim (int): Dimension of input features from backbone
        task_type (str): Type of supervised task ('classification', 'multilabel', 'segmentation')
        num_classes (int): Number of output classes
        hidden_dim (int, optional): Hidden dimension for MLP (default: 512)
    """
    
    def __init__(self, config, input_dim=768, task_type='classification', 
                 num_classes=10, hidden_dim=512):
        super().__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        self.input_dim = input_dim
        
        # Get configuration parameters
        self.aux_weight = getattr(config.auxiliary, 'aux_weight', 1.0)
        self.dropout_rate = getattr(config.auxiliary, 'dropout_rate', 0.1)
        
        # Build the head architecture based on task type
        if task_type in ['classification', 'multilabel']:
            # MLP for classification tasks
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
            # Different output activation based on task
            if task_type == 'classification':
                self.output_activation = nn.LogSoftmax(dim=-1)
                self.loss_fn = nn.NLLLoss()
            else:  # multilabel
                self.output_activation = nn.Sigmoid()
                pos_weight = getattr(config.auxiliary, 'pos_weight', 1.0)
                self.register_buffer('pos_weight', torch.tensor([pos_weight] * num_classes))
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                
        elif task_type == 'segmentation':
            # Decoder for segmentation (simple upsampling)
            self.head = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim // 2),
                nn.GELU(),
                nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)
            )
            self.loss_fn = nn.CrossEntropyLoss()
            
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features, labels=None):
        """
        Forward pass through the supervised head.
        
        Args:
            features: Input features from backbone
                - For classification/multilabel: (Batch * Views, Embed_Dim)
                - For segmentation: (Batch * Views, Embed_Dim, H, W) or (Batch * Views, Embed_Dim)
            labels: Ground truth labels (optional)
                - For classification: (Batch * Views,) class indices
                - For multilabel: (Batch * Views, Num_Classes) multi-hot
                - For segmentation: (Batch * Views, H, W) class indices
        
        Returns:
            If labels provided: (weighted_loss, stats_dict)
            If no labels: predictions
        """
        
        if self.task_type in ['classification', 'multilabel']:
            # Ensure features are 2D
            if features.dim() > 2:
                features = features.mean(dim=[-2, -1])  # Global average pooling
            
            logits = self.head(features)
            predictions = self.output_activation(logits)
            
        else:  # segmentation
            # Reshape features if needed (from ViT patches to spatial)
            if features.dim() == 2:
                # Assume features are flattened patches
                # Need to know patch grid size - for simplicity, assume 14x14 for 224/16
                patch_dim = int(features.shape[-1] ** 0.5)
                if patch_dim ** 2 == features.shape[-1]:
                    features = features.view(features.shape[0], self.input_dim, patch_dim, patch_dim)
                else:
                    # Fallback: reshape to square-ish
                    h = w = int(features.shape[-1] ** 0.5)
                    features = features.view(features.shape[0], self.input_dim, h, w)
            
            logits = self.head(features)
            predictions = F.softmax(logits, dim=1)
        
        # If no labels provided, return predictions
        if labels is None:
            return predictions
        
        # Compute loss
        if self.task_type == 'classification':
            loss = self.loss_fn(predictions, labels)
        elif self.task_type == 'multilabel':
            # For BCEWithLogitsLoss, we need logits not predictions
            loss = self.loss_fn(logits, labels)
        else:  # segmentation
            loss = self.loss_fn(logits, labels)
        
        weighted_loss = loss * self.aux_weight
        
        # Compute metrics
        stats = {
            "aux_loss": loss.item(),
            "aux_weighted_loss": weighted_loss.item(),
            "aux_weight": self.aux_weight
        }
        
        # Add task-specific metrics
        with torch.no_grad():
            if self.task_type == 'classification':
                pred_classes = predictions.argmax(dim=-1)
                accuracy = (pred_classes == labels).float().mean()
                stats["aux_accuracy"] = accuracy.item()
                
            elif self.task_type == 'multilabel':
                preds = (torch.sigmoid(logits) > 0.5).float()
                # Micro-F1
                tp = (preds * labels).sum()
                fp = (preds * (1 - labels)).sum()
                fn = ((1 - preds) * labels).sum()
                f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                stats["aux_f1"] = f1.item()
                
            else:  # segmentation
                pred_classes = logits.argmax(dim=1)
                pixel_accuracy = (pred_classes == labels).float().mean()
                stats["aux_pixel_accuracy"] = pixel_accuracy.item()
        
        return weighted_loss, stats


class MultiTaskSupervisedHead(nn.Module):
    """
    Multi-task supervised head that combines multiple auxiliary heads.
    Useful for testing multiple supervised tasks simultaneously.
    
    Args:
        config: Configuration object
        input_dim (int): Dimension of input features
        heads_config (list): List of dicts with head configurations
    """
    
    def __init__(self, config, input_dim=768, heads_config=None):
        super().__init__()
        
        if heads_config is None:
            # Default configuration: classification + multilabel
            heads_config = [
                {"name": "classification", "task_type": "classification", "num_classes": 10},
                {"name": "multilabel", "task_type": "multilabel", "num_classes": 20}
            ]
        
        self.heads = nn.ModuleDict()
        self.head_configs = {}
        
        for head_cfg in heads_config:
            name = head_cfg["name"]
            task_type = head_cfg.get("task_type", "classification")
            num_classes = head_cfg.get("num_classes", 10)
            hidden_dim = head_cfg.get("hidden_dim", 512)
            
            self.heads[name] = ExampleSupervisedHead(
                config=config,
                input_dim=input_dim,
                task_type=task_type,
                num_classes=num_classes,
                hidden_dim=hidden_dim
            )
            self.head_configs[name] = head_cfg
        
        # Weight for each head (can be configured)
        self.head_weights = {}
        for name in self.heads.keys():
            weight_key = f"{name}_weight"
            self.head_weights[name] = getattr(config.auxiliary, weight_key, 1.0)
    
    def forward(self, features, labels_dict=None):
        """
        Forward pass through all heads.
        
        Args:
            features: Input features
            labels_dict: Dictionary mapping head names to labels
        
        Returns:
            If labels_dict provided: (total_loss, combined_stats)
            If no labels: predictions_dict
        """
        
        if labels_dict is None:
            # Return predictions from all heads
            predictions = {}
            for name, head in self.heads.items():
                predictions[name] = head(features, labels=None)
            return predictions
        
        # Compute losses for each head
        total_loss = 0.0
        combined_stats = {}
        
        for name, head in self.heads.items():
            if name in labels_dict:
                weighted_loss, stats = head(features, labels_dict[name])
                weighted_loss = weighted_loss * self.head_weights[name]
                total_loss += weighted_loss
                
                # Add head-specific stats with prefix
                for stat_name, stat_value in stats.items():
                    combined_stats[f"{name}_{stat_name}"] = stat_value
                
                combined_stats[f"{name}_weight"] = self.head_weights[name]
        
        combined_stats["total_aux_loss"] = total_loss.item()
        
        return total_loss, combined_stats


if __name__ == "__main__":
    # Test the supervised head
    import argparse
    
    class MockConfig:
        class auxiliary:
            aux_weight = 0.5
            dropout_rate = 0.1
            pos_weight = 2.0
            classification_weight = 1.0
            multilabel_weight = 0.8
    
    config = MockConfig()
    
    print("Testing ExampleSupervisedHead with classification task:")
    batch_size = 4
    input_dim = 768
    
    # Test classification head
    head = ExampleSupervisedHead(config, input_dim=input_dim, 
                                 task_type='classification', num_classes=10)
    features = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 10, (batch_size,))
    
    loss, stats = head(features, labels)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Stats: {stats}")
    
    print("\nTesting ExampleSupervisedHead with multilabel task:")
    head = ExampleSupervisedHead(config, input_dim=input_dim,
                                 task_type='multilabel', num_classes=20)
    features = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, 2, (batch_size, 20)).float()
    
    loss, stats = head(features, labels)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Stats: {stats}")
    
    print("\nTesting MultiTaskSupervisedHead:")
    multi_head = MultiTaskSupervisedHead(config, input_dim=input_dim)
    features = torch.randn(batch_size, input_dim)
    labels_dict = {
        "classification": torch.randint(0, 10, (batch_size,)),
        "multilabel": torch.randint(0, 2, (batch_size, 20)).float()
    }
    
    total_loss, stats = multi_head(features, labels_dict)
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Stats keys: {list(stats.keys())}")
