# maskformer model experiments

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

from MERU_utils import lorentz as L
import math
from dataclasses import dataclass

from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation, MaskFormerPreTrainedModel, MaskFormerModel
from transformers.utils import ModelOutput
from PIL import Image
import requests
from transformers import MaskFormerConfig
from torch import Tensor


# Example: ADE20K variant
config = MaskFormerConfig.from_pretrained("facebook/maskformer-swin-base-ade")
model_name = "facebook/maskformer-swin-base-ade"
# config = MaskFormerConfig.from_pretrained(model_name) # bring the config file manually. 

# model_name = "facebook/maskformer-swin-base-coco"


# load MaskFormer fine-tuned on ADE20k semantic segmentation
# image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
# model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")



def semantic_to_instance_masks(label_map):
    # label_map: (H, W) with class IDs [0, 150]
    unique_classes = torch.unique(label_map)
    masks = []
    classes = []
    for c in unique_classes:
        if c.item() == 255:  # ignore label if exists
            continue
        masks.append((label_map == c).float())
        classes.append(c)
    return torch.stack(masks), torch.tensor(classes)

# Example:

def maskformer_label_gen(labels):
    mask_labels, class_labels = [], []
    for label in labels:  # labels: (B, H, W)
        m, c = semantic_to_instance_masks(label)
        mask_labels.append(m)
        class_labels.append(c)
    
    return mask_labels.copy(), class_labels.copy()

import torch

def semantic_to_instance_masks_gpu(label_map, device=None):
    # label_map: (H, W) with class IDs [0, 150]
    if device is None:
        device = label_map.device

    unique_classes = torch.unique(label_map)
    masks = []
    classes = []

    for c in unique_classes:
        if c.item() == 255:  # ignore label if exists
            continue
        masks.append((label_map == c).float().to(device))
        classes.append(c.to(device))

    return torch.stack(masks), torch.tensor(classes, device=device)


def maskformer_label_gen_gpu(labels, device=None):
    mask_labels, class_labels = [], []
    for label in labels:  # labels: (B, H, W)
        m, c = semantic_to_instance_masks_gpu(label, device=device)
        mask_labels.append(m)
        class_labels.append(c)
    
    return mask_labels.copy(), class_labels.copy()





@dataclass
class MeruMaskFormerOutput(ModelOutput):
    pixel_decoder_last_hidden_state: torch.Tensor = None
    transformer_decoder_last_hidden_state: torch.Tensor  = None
    encoder_last_hidden_state: torch.Tensor= None


@dataclass
class MeruMaskFormerForInstanceSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor = None
    class_queries_logits: torch.FloatTensor  = None
    masks_queries_logits: torch.FloatTensor = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: tuple[torch.FloatTensor] = None
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] = None
    transformer_decoder_hidden_states: tuple[torch.FloatTensor] = None
    hidden_states: tuple[torch.FloatTensor] = None
    attentions: tuple[torch.FloatTensor] = None

    

class MeruMaskFormer(MaskFormerForInstanceSegmentation):
    def __init__(self, config = config, num_classes = 151, feat_dim = 64, 
                prototypes = None):
        super().__init__(config) 
        # self.model = self.model
        
        if prototypes is not None:
            self.text_protos = nn.Parameter(prototypes.clone(), requires_grad=False)
        else:
            self.text_protos = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        

        
        # weights for multiplication
        self.textual_alpha = nn.Parameter(torch.tensor(0.3**-1).log(), requires_grad = False) # norm value for TEXT (need checking for 256)  
        self.transformer_alpha =  nn.Parameter(torch.tensor(5**-1).log(), requires_grad = True) # norm value for transformer outputs  
        self.pixel_alpha =  nn.Parameter(torch.tensor(20**-1).log(), requires_grad = True) # norm value for IMAGE pixel decoder output          
        self.mask_alpha =  nn.Parameter(torch.tensor(5**-1).log(), requires_grad = True) # norm value for Mask Embedding outputs
        
        hidden_size =  config.decoder_config.hidden_size # may check the output: should be 256
        self.class_projector = nn.Linear(hidden_size, feat_dim)
    
    def image_encoder(self, pixel_values):
        
        backbone_outputs = self.model.pixel_level_module(pixel_values)
        transformer_module_output = self.model.transformer_module(backbone_outputs[0])
        
        outputs = MeruMaskFormerOutput(
            pixel_decoder_last_hidden_state = backbone_outputs[1],
            transformer_decoder_last_hidden_state  = transformer_module_output[0]
        )
        return outputs # return euclidean outputs
    
    def get_logits(self, outputs) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        pixel_embeddings = outputs.pixel_decoder_last_hidden_state
        # get the auxiliary predictions (one for each decoder's layer)
        auxiliary_logits: list[str, Tensor] = []
        transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state
        # euclidean operation
        classes = self.class_predictor(transformer_decoder_hidden_states) # stage 1: use prototypes here. should be shape of #no_of_classes, 256 (d)
        class_queries_logits = classes
        # get the masks
        mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states) # euclidean operation # not good idea to add another hyperbolic layer 
        # sum up over the channels
        masks_queries_logits = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings) # euclidean operation
        return class_queries_logits, masks_queries_logits, auxiliary_logits
    
    def get_hyperbolic_logits(self, outputs):
        pixel_embeddings = outputs.pixel_decoder_last_hidden_state
        l_p_d_f = L.exp_map0(pixel_embeddings.permute(0,2,3,1) * self.pixel_alpha.exp())  # lorentz pixel decoder features: shape of BS, h,w, feature-dim(256)
        
        transformer_decoder_hidden_states = outputs.transformer_decoder_last_hidden_state  # shape of bs, 100, 256(d) # a 3d matrix
        trans_dec_proj = self.class_projector(transformer_decoder_hidden_states)
        l_t_d  = L.exp_map0(trans_dec_proj * self.transformer_alpha.exp()) # lorentz transformer_decoder_hidden_states # a 3d matrix  
        # Hyperbolic operation
        l_tp = L.exp_map0(self.text_protos* self.textual_alpha.exp()) # lorentz text prototypes # shape of class_number, 256 (d) # a 2d matrix
        
        # angle: text to class logits
        classes_agl =  L.oxy_angle_full_gnr(l_t_d, l_tp) # angle with class text prototypes (bs, 100, class_number)
        # alternatively distance
        classes_dist = L.pairwise_dist(l_t_d, l_tp)
        
        
        
        classes = classes_agl + 0.5*classes_dist # combination of the distance and angles
        
        
        class_queries_logits = - classes/0.08 # the minus sign: as lower means better aligned ; reverse if for logit concept: higher is better 
        # requires scaling only
        
        # get the masks in hyperbolic operation
        
        

        
        mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states) # euclidean operation # not good idea to add another hyperbolic layer 
        # # #Improvmement Idea : Normalize the mask_embeddings Before the exponential Map
        # mask_embeddings = F.normalize(mask_embeddings,  p=2, dim=-1) # in that case make the self.mask_alpha = 1 . log..
        
        l_m_e = L.exp_map0(mask_embeddings*self.mask_alpha.exp()) # lorentz mask embedding
        
        angle_val, dist_val =  L.oxy_angle_mask(l_p_d_f, l_m_e) # raw value of angle and the distance (lower the better for each classes)
        
#         #angles
        
#         #mask_q_l range (-pi to 0); to pass it via sigmoid: shift and scale
        
        # uncomment for the angle based computation
        masks_queries_logits_a = - angle_val # or dist_val or weighted sum # reverse it as higher means better; for angle lower means better
        masks_queries_logits_a =  (masks_queries_logits_a + 0.15)/ 0.02 # might tune the shift (+ 0.15) and scale (1/0.04) parameters 180/3.14 * shift degree for penalty

        # distance (reversing and then shift and scale)
        masks_queries_logits_d = - dist_val # or dist_val or weighted sum # reverse it as higher means better; for angle lower means better
        masks_queries_logits_d =  (masks_queries_logits_d + 1)/ 0.05 # might tune the shift (+ 0.8) and scale (1/0.05) parameters 180/3.14 * shift degree for penalty
        
        
        
        
        masks_queries_logits = masks_queries_logits_a + masks_queries_logits_d
        
        
        masks_queries_logits = masks_queries_logits.permute(0,3,1,2) 

        return class_queries_logits, masks_queries_logits, None # just making it three returns as necessary
    
    def forward(self,
        pixel_values: Tensor,
        mask_labels=  None,
        class_labels= None,
        pixel_mask= None):
        
        
        encoder_val =  self.image_encoder(pixel_values)
        class_queries_logits, masks_queries_logits, auxiliary_logits = self.get_hyperbolic_logits(encoder_val)
        
        
        if mask_labels is not None and class_labels is not None:
            loss_dict: dict[str, Tensor] = self.get_loss_dict(
                masks_queries_logits, class_queries_logits, mask_labels, class_labels, auxiliary_logits
            )
            loss = self.get_loss(loss_dict)
        else: 
            loss = None

        return MeruMaskFormerForInstanceSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
            pixel_decoder_last_hidden_state = encoder_val.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state = encoder_val.transformer_decoder_last_hidden_state
            
        )
        
