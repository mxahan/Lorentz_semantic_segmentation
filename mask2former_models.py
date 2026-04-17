# custom pytorch maskformer 2 model
# maskformer model experiments

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

from MERU_utils import lorentz as L
import math
from dataclasses import dataclass

from transformers.utils import ModelOutput
from PIL import Image
import requests
from transformers import Mask2FormerConfig
from torch import Tensor


from transformers import Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerMaskPredictor, Mask2FormerLoss, Mask2FormerHungarianMatcher


# Example: ADE20K variant
model_name =  "facebook/mask2former-swin-large-ade-semantic"
config = Mask2FormerConfig.from_pretrained(model_name)



class CustomMask2FormerLoss(Mask2FormerLoss):
    """
    A subclass of MaskFormerLoss for custom modifications.
    You can override methods like `forward` or add new loss components.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add your custom initialization here later

    def forward(self, *args, **kwargs):
        # For now, just call the base class forward
        return super().forward(*args, **kwargs)

    


@dataclass
class Mask2FormerForUniversalSegmentationOutput(ModelOutput):
    loss: torch.FloatTensor = None
    class_queries_logits: torch.FloatTensor = None
    masks_queries_logits: torch.FloatTensor = None
    auxiliary_logits: list[dict[str, torch.FloatTensor]] = None
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state: torch.FloatTensor = None
    encoder_hidden_states: tuple[torch.FloatTensor] = None
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] = None
    transformer_decoder_hidden_states: torch.FloatTensor = None
    attentions: tuple[torch.FloatTensor] = None
    

@dataclass
class Mask2FormerModelOutput(ModelOutput):
    encoder_last_hidden_state: torch.FloatTensor = None
    pixel_decoder_last_hidden_state: torch.FloatTensor = None
    transformer_decoder_last_hidden_state:  torch.FloatTensor = None
    encoder_hidden_states: tuple[torch.FloatTensor] = None
    pixel_decoder_hidden_states: tuple[torch.FloatTensor] = None
    transformer_decoder_hidden_states: tuple[torch.FloatTensor] = None
    transformer_decoder_intermediate_states: tuple[torch.FloatTensor] = None
    masks_queries_logits: tuple[torch.FloatTensor] = None
    attentions: tuple[torch.FloatTensor] = None

    

class MeruMask2Former(Mask2FormerForUniversalSegmentation):
    def __init__(self, config = config, num_classes = 151, feat_dim = 64, 
                prototypes = None):
        super().__init__(config)
        
        
        if prototypes is not None:
            self.text_protos = nn.Parameter(prototypes.clone(), requires_grad=False)
        else:
            self.text_protos = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        
        # weights for multiplication
        self.textual_alpha = nn.Parameter(torch.tensor(0.3**-1).log(), requires_grad = False) # norm value for TEXT (need checking for 256)  
        self.transformer_alpha =  nn.Parameter(torch.tensor(3**-1).log(), requires_grad = True) # norm value for transformer outputs  # original value was 5
    
        self.class_projector = nn.Linear(config.hidden_dim, feat_dim)
        
        # You can add your custom initialization here later
        
    def image_encoder(self, pixel_values,
            pixel_mask,
            output_hidden_states,
            output_attentions,
            return_dict=True,):
        
        outputs = self.model(pixel_values,pixel_mask,
            output_hidden_states,
            output_attentions,
            return_dict=True)
        
        return outputs
        
    def forward(self,
        pixel_values: Tensor,
        mask_labels=  None,
        class_labels= None,
        pixel_mask= None, 
        output_hidden_states: bool | None = None
        ) -> Mask2FormerForUniversalSegmentationOutput:
        
        
        output_attentions = False
  
        # outputs = self.image_encoder(
        #     pixel_values=pixel_values,
        #     pixel_mask=pixel_mask,
        #     output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
        #     output_attentions=output_attentions,
        #     return_dict=True,
        # )
    
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states or self.config.use_auxiliary_loss,
            output_attentions=output_attentions,
            return_dict=True,
        )

        loss, loss_dict, auxiliary_logits = None, None, None
        class_queries_logits = ()

        for decoder_output in outputs.transformer_decoder_intermediate_states:
            # this operation needs to be replaced with the hyperbolic
            ### regular euclidean only operation
            # class_prediction = self.class_predictor(decoder_output.transpose(0, 1)) # need to replace this line in particular9
            
            
            ### hyperbolic lines
            trans_dec_proj = self.class_projector(decoder_output.transpose(0, 1))
            #  #  # Improvmement Idea : Normalize the trans_dec_proj Before the exponential Map # may be dont need it thought!
            # trans_dec_proj = F.normalize(trans_dec_proj,  p=2, dim=-1) # in that case make the self.mask_alpha = 1 . log..
            
            l_t_d  = L.exp_map0(trans_dec_proj * self.transformer_alpha.exp()) # lorentz transformer_decoder_hidden_states # a 3d matrix  
            l_tp = L.exp_map0(self.text_protos* self.textual_alpha.exp()) # lorentz text prototypes # shape of class_number, 256 (d) # a 2d matrix
            # angle: text to class logits
            classes_agl =  L.oxy_angle_full_gnr(l_t_d, l_tp) # angle with class text prototypes (bs, 100, class_number)
            # alternatively distance
            classes_dist = L.pairwise_dist(l_t_d, l_tp)
            
            classes = classes_agl + 0.5*classes_dist # combination of the distance and angles
            
            class_prediction = - classes/0.08 
            ###
            
            
            
            class_queries_logits += (class_prediction,)
            
            

        masks_queries_logits = outputs.masks_queries_logits

        auxiliary_logits = self.get_auxiliary_logits(class_queries_logits, masks_queries_logits)

        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_logits,
            )
            loss = self.get_loss(loss_dict)

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = outputs.encoder_hidden_states
            pixel_decoder_hidden_states = outputs.pixel_decoder_hidden_states
            transformer_decoder_hidden_states = outputs.transformer_decoder_hidden_states


        output = Mask2FormerForUniversalSegmentationOutput(
            loss=loss,
            class_queries_logits=class_queries_logits[-1],
            masks_queries_logits=masks_queries_logits[-1],
            auxiliary_logits=auxiliary_logits,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            pixel_decoder_last_hidden_state=outputs.pixel_decoder_last_hidden_state,
            transformer_decoder_last_hidden_state=outputs.transformer_decoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            pixel_decoder_hidden_states=pixel_decoder_hidden_states,
            transformer_decoder_hidden_states=transformer_decoder_hidden_states,
            attentions=outputs.attentions,
        )
        return output

    
class MeruMask2Predictor(Mask2FormerMaskPredictor):
    def __init__(self, hidden_size, num_heads, mask_feature_size):
        """
        A drop-in replacement for Mask2FormerMaskPredictor
        that preserves the original architecture & weights.
        """
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mask_feature_size=mask_feature_size,
        )
        
        self.pixel_alpha =  nn.Parameter(torch.tensor(20**-1).log(), requires_grad = True) # norm value for IMAGE pixel decoder output          
        self.mask_alpha =  nn.Parameter(torch.tensor(5**-1).log(), requires_grad = True) # norm value for Mask Embedding outputs # else 5
     
        
    def forward(
        self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None
    ):
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))
        #  #  # Improvmement Idea : Normalize the mask_embedding Before the exponential Map
        # mask_embeddings = F.normalize(mask_embeddings,  p=2, dim=-1) # in that case make the self.mask_alpha = 1 . log..

        
        # the following line needs to be replaced by the 
        # outputs_mask = torch.einsum("bqc, bchw -> bqhw", mask_embeddings, pixel_embeddings) # commented euclidean operations
        
        ### Hyperbolic Operations
        
        l_p_d_f = L.exp_map0(pixel_embeddings.permute(0,2,3,1) * self.pixel_alpha.exp()) 
        l_m_e = L.exp_map0(mask_embeddings*self.mask_alpha.exp()) # lorentz mask embedding
        
    
        angle_val, dist_val =  L.oxy_angle_mask(l_p_d_f, l_m_e) # raw value of angle and the distance (lower the better for each classes)
        
        #mask_q_l range (-pi to 0); to pass it via sigmoid: shift and scale
        
        # uncomment for the angle based computation
        masks_queries_logits_a = - angle_val # or dist_val or weighted sum # reverse it as higher means better; for angle lower means better
        masks_queries_logits_a =  (masks_queries_logits_a + 0.15)/ 0.02 # might tune the shift (+ 0.15) and scale (1/0.04) parameters 180/3.14 * shift degree for penalty

        # distance (reversing and then shift and scale)
        masks_queries_logits_d = - dist_val # or dist_val or weighted sum # reverse it as higher means better; for angle lower means better
        masks_queries_logits_d =  (masks_queries_logits_d + 1)/ 0.05 # might tune the shift (+ 0.8) and scale (1/0.05) parameters 180/3.14 * shift degree for penalty
        
        
        masks_queries_logits = masks_queries_logits_a + masks_queries_logits_d
        
        
        outputs_mask = masks_queries_logits.permute(0,3,1,2) 

            
        ### Hyperbolic operations
        
        

        attention_mask = nn.functional.interpolate(
            outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False
        )

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        return outputs_mask, attention_mask
    



