import torch
import torch.nn as nn
import torchvision

class FastRCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(FastRCNN, self).__init__()
        
        # Backbone: VGG19 without fully connected layers
        vgg19 = VGG19(num_classes)
        self.features = nn.Sequential(*list(vgg19.children())[:-1])
        
        # RoI pooling layer
        self.roi_pool = torchvision.ops.RoIPool(output_size=(7, 7), spatial_scale=1/16)
        
        # Fully connected layers
        # input size: 512 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # output size: 4096
        # Classifier
        self.classifier = nn.Linear(4096, num_classes)
        
        # Bounding box regressor
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)
    
    def forward(self, x, rois_list):
        x = self.features(x)
        
        class_scores_list = []
        
        bbox_deltas_list = []
        
        for roi in rois_list:
            # RoI pooling
            pooled_features = self.roi_pool(x, roi)
            
            # Apply fully connected layers
            x = pooled_features.view(pooled_features.size(0), -1)
            x = self.fc(x)
            
            # Classify the RoIs
            class_scores = self.classifier(x)
            
            # Bounding box regression
            bbox_deltas = self.bbox_regressor(x)
            class_scores_list.append(class_scores)
            bbox_deltas_list.append(bbox_deltas)
            
        
        return class_scores_list, bbox_deltas_list
        
        # # RoI pooling
        # pooled_features = self.roi_pool(x, rois)
        
        # # Apply fully connected layers
        # x = pooled_features.view(pooled_features.size(0), -1)
        # x = self.fc(x)
        
        # # Classify the RoIs
        # class_scores = self.classifier(x)
        
        # # Bounding box regression
        # bbox_deltas = self.bbox_regressor(x)
        
        return class_scores, bbox_deltas
