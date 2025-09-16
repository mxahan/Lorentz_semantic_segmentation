import torch
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

# Load pretrained DeepLabV3 with a ResNet-101 backbone
## Pytorch Version

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes = None):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, atrous_rates=[12, 24, 36]),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            # nn.Conv2d(256, num_classes, kernel_size=1)
        )

from MERU_utils import lorentz as L

from torchvision.models.segmentation.deeplabv3 import ASPP

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(CustomDeepLabV3, self).__init__()
 
        # ----- #
        # Load pretrained DeepLabV3 with ResNet-50 backbone
        
        # citiscape verion
        # model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        # cocostaff model (model zoo)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', weights='COCO_WITH_VOC_LABELS_V1')
        # or any of these variants
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
     
        # Retain backbone
        self.backbone = model.backbone
        # Replace classifier head with custom head (new parameters)
        self.head = DeepLabHead(in_channels=2048, num_classes = num_classes)
        # Add final classifier layer separately (untrained)
        self.relu = nn.ReLU()
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def image_encoder(self, x: torch.Tensor,  project: False):
        """
        Args: images: x: Image batch in BCHW format
        Returns: Batch of image features of shape  BCHW, BCHW.
        """
        
        input_shape = x.shape[-2:]
        features = self.backbone(x)['out']  # Backbone features
        x = self.head(features)             # Custom head (up to pre-ReLU)
        out = self.classifier(self.relu(x))            # Final classifier
        out = nn.functional.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out
        
    def forward(self, x, labels = None, criterion = F.cross_entropy, mask = None):
        
        out = self.image_encoder(x)
        # return out, x  # return both output and pre-classifier features
        if labels is not None:
            ignore_index = 255  # or -100 depending on dataset
            if mask is not None:
                # Mark ignored positions where mask==1
                labels = labels.masked_fill(mask.bool(), ignore_index)

            loss = criterion(out, labels, ignore_index=ignore_index)
            return {"loss": loss, "logits": out}
        else:
            return {"logits": out}
        
# Euclidean mdoel

class ProtoDeepLabV3(CustomDeepLabV3):
    def __init__(self, num_classes, feat_dim, prototypes=None):
        super().__init__(num_classes=num_classes)

        # Change classifier to output feature embeddings instead of logits
        self.classifier = nn.Conv2d(256, feat_dim, kernel_size=1)

        # Define prototypes: fixed if provided, learnable otherwise
        if prototypes is not None:
            self.text_protos = nn.Parameter(prototypes.clone(), requires_grad=False)
        else:
            self.text_protos = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)

        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def forward(self, x, labels=None, criterion=F.cross_entropy, mask=None):
        # === Keep image_encoder intact ===
        feats = self.image_encoder(x)  # (B, feat_dim, H, W) now because classifier changed

        # Normalize features and prototypes for cosine similarity
        feats_norm = F.normalize(feats, dim=1)          # (B, feat_dim, H, W)
        protos_norm = F.normalize(self.text_protos, dim=1)  # (num_classes, feat_dim)

        # Compute per-pixel similarity with prototypes
        logits = torch.einsum("bfhw,cf->bchw", feats_norm, protos_norm)

        # If labels provided, compute loss
        if labels is not None:
            ignore_index = 255
            if mask is not None:
                labels = labels.masked_fill(mask.bool(), ignore_index)

            loss = criterion(logits, labels, ignore_index=ignore_index)
            return {"loss": loss, "logits": logits, "feats": feats}
        else:
            return {"logits": logits, "feats": feats}



## more models from MMSEG


class MeruDeepLabV3(CustomDeepLabV3):
    def __init__(self, num_classes, 
                 feat_dim : int = 64, prototypes = None, 
                 curv_init : float = 1, device = torch.device('cuda'), 
                 learn_curv: bool = False, 
                 entail_weight : float = 0.0):
        super(MeruDeepLabV3, self).__init__(num_classes)
        self.feats = nn.Conv2d(256, feat_dim, kernel_size=1)
                
        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())
        
        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.textual_alpha = nn.Parameter(torch.tensor(0.6**-1).log(), requires_grad = False) # norm value for TEXT
        self.image_alpha =  nn.Parameter(torch.tensor(4**-1).log(), requires_grad = True) # norm value for IMAGE
        
        # Initialize learnable text class prototypes
        if prototypes is not None:
            self.text_protos = nn.Parameter(prototypes.clone(), requires_grad = False)  # learnable (num_classes, feat_dim)
        else:
            self.text_protos = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
        self.device = device
        
        self._curv_minmax = {
            "max": math.log(curv_init * 10),
            "min": math.log(curv_init / 10)}
        self.entail_weight = entail_weight


    def image_encoder(self, x: torch.Tensor,  project: bool):
        """
        Args: images: x: Image batch in BCHW format
        Returns: Batch of image features of shape  BCHW, BCHW.
        """
        input_shape = x.shape[-2:]
        features = self.backbone(x)['out']  # Backbone features
        x = self.relu(self.head(features))  # Custom head (up to pre-ReLU)
        # Final classifier
        out = self.classifier(x)            
        out = nn.functional.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        # features
        feat_out = self.feats(x)
        feat_out = nn.functional.interpolate(feat_out, size=input_shape, mode='bilinear', align_corners=False)
        
        if project:
            feat_out = feat_out.permute(0,2,3,1) * self.image_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                feat_out = L.exp_map0(feat_out, self.curv.exp()) # equivalent to x_space of the lorentz
        
        return out, feat_out  # return both output and pre-classifier features
    
    def text_encoder(self, project: bool):

        if project:
            text_feats = self.text_protos * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.curv.exp())
        return text_feats
    
    def forward(self, x, labels = None, mask = None):
        '''
        x: tensor (Bs, 3, H, W)
        labels: (Bs, H, W)
        mask: (Bs, H, W), binary (1 = ignore, 0 = include)
        '''
        image_feats = self.image_encoder(x, True) # Lorent/Euclidean output
        # if True return BS, H, W, D tensor
        text_feats =  self.text_encoder(True) # (Class_num, feat_dim)
        
        self.image_alpha.data = torch.clamp(self.image_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()
        
        with torch.autocast(self.device.type, dtype=torch.float32):
            # ## Compute logits for Supervised loss.
            if labels is None: 
                logits = -L.pairwise_dist(image_feats, text_feats, _curv)
                return {'logits' : logits}
            
            # image_logits_1 = -L.pairwise_dist(image_feats, text_feats, _curv) # (Bs, H, W, Class_num)
            # ## Aperture angle
            # _angle_1 = L.oxy_angle(text_feats[labels], image_feats, _curv)  
            
            # replacement of earlier line: efficient version
            _angle, image_logits, _aperture = L.oxy_angle_modified(image_feats, text_feats, labels = labels, curv = _curv) 
            image_logits = - image_logits # taking - of the distance
            
            # logit temperature
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()
            # supervised loss
            
            # === Apply mask if provided ===
            ignore_index = 255 # or -100, depending on your pipeline

            if mask is not None:
                labels = labels.masked_fill(mask.bool(), ignore_index)
                            # entailment masking: ignore masked positions
                entail_mask = ~mask.bool()  # True where valid
                _angle = _angle[entail_mask]
                _aperture = _aperture[entail_mask]


            # supervised loss
            sup_loss = nn.functional.cross_entropy(
                _scale * image_logits.permute(0, 3, 1, 2),
                labels,
                ignore_index=ignore_index,
            )

            # sup_loss = nn.functional.cross_entropy(_scale* image_logits.permute(0,3, 1, 2), labels) 
            # Aperture angle
            # _angle = L.oxy_angle(text_feats[labels], image_feats, _curv) # calculated earlier    
            # _aperture = L.half_aperture(text_feats[labels], _curv) # bs, H, W # calculated earlier
            # entailment_loss = torch.clamp(_angle - _aperture, min=0).mean() 
            
            if _angle.numel() > 0:  # safeguard against all-masked case
                entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()
            else:
                entailment_loss = torch.tensor(0.0, device=x.device, requires_grad=True)



            loss = sup_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss
 
        return {
            "loss": loss,
            "logging": {
                "supervised_loss": sup_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }