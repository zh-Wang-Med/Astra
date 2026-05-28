import torch.nn as nn
from transformers import BertModel, BertTokenizer
from merlin import Merlin_classify
from merlin.models import inflate
import ipdb
import torch
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
from transformer_maskgit import CTViT
import torch.nn.functional as F
def l2norm(t):
    return F.normalize(t, dim = -1)
def sanitize_report(report):
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))
class CTCLIP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.visual_transformer = CTViT(
            dim=512, codebook_size=8192, image_size=480, patch_size=20,
            temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
            dim_head=32, heads=8
        )
        self.to_visual_latent = nn.Linear(294912, 512, bias = False)
        if self.args.load_pretrain:
            state_dict = torch.load(self.args.pretrain_path)
            self.load_state_dict(state_dict=state_dict, strict=False)

    def forward(self, image):
        enc_image= self.visual_transformer(image, return_encoded_tokens=True) # b,24,24,24,512
        # ipdb.set_trace()
        enc_image = torch.mean(enc_image, dim=1) # b,24,24,512
        enc_image = enc_image.view(enc_image.shape[0], -1)  # b, 294912
        image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image
        image_latents = self.to_visual_latent(image_embeds)
        image_latents = l2norm(image_latents)
        return image_latents


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings

class labeler(nn.Module):
    def __init__(self, args, p):
        super().__init__()

        self.p = p
        self.args = args

        if self.args.dataset == 'ctrate':
            self.label_names = [
                'Medical_material', 'Arterial_wall_calcification', 'Cardiomegaly', 'Pericardial_effusion',
                'Coronary_artery_wall_calcification', 'Hiatal_hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung_nodule', 'Lung_opacity', 'Pulmonary_fibrotic_sequela',
                'Pleural_effusion', 'Mosaic_attenuation_pattern', 'Peribronchial_thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular_septal_thickening'
            ]
            disease_num = 18
        if self.args.dataset == 'radchest':
            self.label_names = [
                'Medical_material', 'Cardiomegaly', 'Pericardial_effusion',
                'Hiatal_hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung_nodule', 'Lung_opacity', 'Pulmonary_fibrotic_sequela',
                'Pleural_effusion', 'Peribronchial_thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular_septal_thickening', 'calcification'
            ]
            disease_num = 16
        if self.args.dataset == 'merlin':
            self.label_names = [
                'submucosal_edema', 'renal_hypodensities', 'aortic_valve_calcification',
                'coronary_calcification', 'thrombosis', 'metastatic_disease',
                'pancreatic_atrophy', 'renal_cyst', 'osteopenia',
                'surgically_absent_gallbladder', 'atelectasis', 'abdominal_aortic_aneurysm',
                'anasarca', 'hiatal_hernia', 'lymphadenopathy',
                'prostatomegaly', 'biliary_ductal_dilation', 'cardiomegaly',
                'splenomegaly', 'hepatomegaly', 'atherosclerosis',
                'ascites', 'pleural_effusion', 'hepatic_steatosis',
                'appendicitis', 'gallstones', 'hydronephrosis',
                'bowel_obstruction', 'free_air', 'fracture'
            ]
            disease_num = 30
        if self.args.dataset == 'rsna_pe':
            self.label_names = ['leftsided_pe','rightsided_pe','central_pe','chronic_pe']
            disease_num = 4

        if self.args.vision_encoder == "merlin":
            self.image_encoder = Merlin_classify(ImageEmbedding=True)
            if self.args.merlin_pretrain_path is not None:
                meriln_state_dict = torch.load(self.args.merlin_pretrain_path, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))
                self.image_encoder.load_state_dict(state_dict=meriln_state_dict, strict=True)
            hidden_size = 2048
            intermediate_size = 768
            if args.text_help == False:
                if self.args.linear_number == 1:
                    self.classification_heads = nn.Sequential(
                            nn.ReLU(),
                            nn.Dropout(self.p),
                            nn.Linear(hidden_size, disease_num, bias = True)
                        ) 
                if self.args.linear_number == 2:
                    self.classification_heads = nn.Sequential(
                        nn.Linear(hidden_size, intermediate_size),
                        nn.ReLU(),
                        nn.Linear(intermediate_size, disease_num, bias = True)
                    ) 
            else:
                self.text_proj = nn.Sequential(
                        nn.Linear(4096, hidden_size),
                        nn.ReLU(),
                    )
                self.classification_heads = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, disease_num, bias = True)
                    ) 
        
        if self.args.vision_encoder == "ct_clip":
            self.image_encoder = CTCLIP(self.args)
            hidden_size = 512
            intermediate_size = 256
            if args.text_help == False:
                if self.args.linear_number == 1:
                    self.classification_heads = nn.Sequential(
                            nn.ReLU(),
                            nn.Dropout(self.p),
                            nn.Linear(hidden_size, disease_num, bias = True)
                        ) 
                if self.args.linear_number == 2:
                    self.classification_heads = nn.Sequential(
                        nn.Linear(hidden_size, intermediate_size),
                        nn.ReLU(),
                        nn.Linear(intermediate_size, disease_num, bias = True)
                    ) 
            else:
                self.text_proj = nn.Sequential(
                        nn.Linear(4096, hidden_size),
                        nn.ReLU(),
                    )
                self.classification_heads = nn.Sequential(
                        nn.Linear(hidden_size*2, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, disease_num, bias = True)
                )
        # ipdb.set_trace()


        if self.args.visual_frozen:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

            
    def forward(self, images):
        final_hidden = self.image_encoder(images).squeeze(0)
        
        # Apply each classification head
        predictions = self.classification_heads(final_hidden)
        
        return predictions
    
    def forward_with_text(self, images,text_embedding):
        # ipdb.set_trace()
        if self.args.text_help:
            text_embedding = self.text_proj(text_embedding)
            final_hidden = self.image_encoder(images).squeeze(0)
            # ipdb.set_trace()
            fused = torch.cat([final_hidden, text_embedding], dim=-1)  # (B, 4096)
            predictions = self.classification_heads(fused)
        else:
            final_hidden = self.image_encoder(images).squeeze(0)
            predictions = self.classification_heads(final_hidden)
        # ipdb.set_trace()
        return predictions


