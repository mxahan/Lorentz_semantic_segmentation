## Custom and MERU segformer model

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerDecodeHead, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from MERU_utils import lorentz as L
import math


'''
pretrained_name
nvidia/segformer-b0-finetuned-ade-512-512

nvidia/segformer-b1-finetuned-ade-512-512

nvidia/segformer-b2-finetuned-ade-512-512

nvidia/segformer-b3-finetuned-ade-512-512

nvidia/segformer-b4-finetuned-ade-512-512

nvidia/segformer-b5-finetuned-ade-512-512
'''

class CustomSegformer(nn.Module):
    def __init__(self, backbone_name="nvidia/segformer-b4-finetuned-ade-512-512", num_classes=151):
        super().__init__()     
        # Load full pretrained model (encoder + decoder)
        base_model = SegformerForSemanticSegmentation.from_pretrained(backbone_name)     
        # Encoder (SegformerModel)
        self.encoder = base_model.segformer  # SegformerModel (encoder only)    
        # Decoder head
        self.decode_head = base_model.decode_head    
        # Replace classifier layer with new number of classes
        in_channels = self.decode_head.classifier.in_channels
        self.decode_head.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        

    def image_encoder(self, pixel_values: torch.Tensor,  project: False):      
        # 1. Encoder
        encoder_outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states  # list of [B, C, H, W] feature maps
        # 2. Decoder
        logits = self.decode_head(hidden_states)  # [B, num_classes, H/4, W/4]
        # 3. Upsample to input image size
        upsampled_logits = F.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,)
        
        return upsampled_logits
        
                
    def forward(self, pixel_values, labels=None, mask= None):
        
        upsampled_logits = self.image_encoder(pixel_values)
        # 4. Loss computation
        ignore_index = 0 # or -100, depending on your pipeline
        if labels is not None:
            if mask is not None:
                # Convert binary mask -> ignore positions (mask==1 ignored, mask==0 included)
                labels = labels.masked_fill(mask.bool(), ignore_index)

            loss = F.cross_entropy(
                upsampled_logits,
                labels,
                ignore_index=ignore_index
            )
            return {"loss": loss, "logits": upsampled_logits}

        else:
            return {"logits": upsampled_logits}
        
# Euclidean model 

class ProtoSegformer(CustomSegformer):
    def __init__(self, backbone_name="nvidia/segformer-b3-finetuned-ade-512-512", num_classes=151, feat_dim=256, prototypes=None):
        super().__init__(backbone_name=backbone_name, num_classes=num_classes)
        
        # Replace classifier to output embeddings (feat_dim) instead of class logits
        in_channels = self.decode_head.classifier.in_channels
        self.decode_head.classifier = nn.Conv2d(in_channels, feat_dim, kernel_size=1)

        # Define prototypes: fixed if provided, learnable otherwise
        if prototypes is not None:
            self.text_protos = nn.Parameter(prototypes.clone(), requires_grad=False)
        else:
            self.text_protos = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)

        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def forward(self, pixel_values, labels=None, mask=None):
        # === Keep original image_encoder intact ===
        feats = self.image_encoder(pixel_values)  # (B, feat_dim, H, W) because classifier changed above

        # Normalize features and prototypes for cosine similarity
        feats_norm = F.normalize(feats, dim=1)            # (B, feat_dim, H, W)
        protos_norm = F.normalize(self.text_protos, dim=1)  # (num_classes, feat_dim)

        # Compute per-pixel similarity (logits): (B, num_classes, H, W)
        logits = torch.einsum("bfhw,cf->bchw", feats_norm, protos_norm)

        # Loss computation if labels provided
        if labels is not None:
            ignore_index = 255
            if mask is not None:
                labels = labels.masked_fill(mask.bool(), ignore_index)

            loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
            return {"loss": loss, "logits": logits, "feats": feats}
        else:
            return {"logits": logits, "feats": feats}



# Segformer MERU model

class MeruSegformer(CustomSegformer):
    def __init__(self, num_classes, 
                 feat_dim : int = 64, prototypes = None, 
                 curv_init : float = 1, device = torch.device('cuda'), 
                 learn_curv: bool = False, 
                 entail_weight : float = 0.0):
        super(MeruSegformer, self).__init__(backbone_name="nvidia/segformer-b4-finetuned-ade-512-512", num_classes=feat_dim)    
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


    def image_encoder(self, pixel_values: torch.Tensor,  project: bool):
        """
        Args: images: pixel_values: Image batch in BCHW format
        Returns: Batch of image features of shape  BCHW, BCHW.
        """
        # 1. Encoder
        encoder_outputs = self.encoder(pixel_values, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states  # list of [B, C, H, W] feature maps
        # 2. Decoder
        logits = self.decode_head(hidden_states)  # [B, num_classes, H/4, W/4]
        # 3. Upsample to input image size
        feat_out = F.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,)
        
        
        if project:
            feat_out = feat_out.permute(0,2,3,1) * self.image_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                feat_out = L.exp_map0(feat_out, self.curv.exp()) # equivalent to x_space of the lorentz
        
        return feat_out  # return both output and pre-classifier features
    
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
                return {'logits': logits}
            
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
                # supervised CE masking
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
            
            
            # entailment loss (masked if needed)
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
    