'''
 @FileName    : interhuman_diffusion_phys.py
 @EditTime    : 2024-04-15 15:12:19
 @Author      : Buzhen Huang
 @Email       : buzhenhuang@outlook.com
 @Description : 
'''

import torch
import os
from torch import nn
from model.interhuman_diffusion_phys import interhuman_diffusion_phys
import scipy.ndimage.filters as filters
from model.utils import *
from model.blocks import *
import clip

class ControlNet(nn.Module):
    def __init__(self, frame_length=16, **kwargs):
        super(ControlNet, self).__init__()
        self.cfg_weight = 0
        self.num_frames = frame_length
        self.latent_dim = 256
        self.ff_size = self.latent_dim * 2
        self.num_layers = 4
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = 'gelu'
        self.hint_feats = 256
        self.time_embed_dim = 1024
        self.feature_emb_dim = 256

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        
        # CLIP
        clip_model, _ = clip.load("/root/autodl-fs/ViT-B-32.pt", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        self.clip_ln = nn.LayerNorm(512)
        # Hint Block
        self.text_project = nn.Linear(512, self.latent_dim)
        nn.init.zeros_(self.text_project.weight)
        nn.init.zeros_(self.text_project.bias)
            
        # Hint Block
        self.first_zero_linear = nn.Linear(self.hint_feats, self.latent_dim)
        nn.init.zeros_(self.first_zero_linear.weight)
        nn.init.zeros_(self.first_zero_linear.bias)

        self.mid_zero_linear = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)])
        for m in self.mid_zero_linear:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # Trans Encoder
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.blocks.append(TransformerBlock(num_heads=self.num_heads,latent_dim=self.latent_dim, dropout=self.dropout, ff_size=self.ff_size))
            
        # # Load part of the pretrained model for self.blocks
        # base_model_path = '/root/autodl-tmp/Hi4D/out/mask_invalid_feature/03.21-08h27m16s/trained model/best_reconstruction_epoch012_210.783051.pkl'
        # state_dict = torch.load(base_model_path)['model']
        # for i in range(self.num_layers):
        #     self.blocks[i].load_state_dict({k: v for k, v in state_dict.items() if 'blocks.{}.{}'.format(i, k) in k}, strict=False)
        # print("Loaded base model encoder from", base_model_path)

    def text_process(self, data):
        use_label = True # use label or use sentence
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        text1 = data['text1']
        text2 = data['text2']
        raw_text_batch = text1 + text2

        with torch.no_grad():
            device = next(self.clip_transformer.parameters()).device
            text = clip.tokenize(raw_text_batch, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out) # [B, T, 512]
        text_cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)] # [B, 512]
        text_cond = self.text_project(text_cond)
        text_cond1 = text_cond[:batch_size]
        text_cond2 = text_cond[batch_size:]
        text_cond = torch.stack([text_cond1, text_cond2], dim=1) # [batch_size, 2, 256]
        return text_cond
    
    def forward(self, h_prev, emb, img_info, data, mean):
        batch_size, frame_length, agent_num = data['features'].shape[:3]
        
        text_cond = self.text_process(data) # [batch_size, agent_num, 256]
        text_cond = text_cond.unsqueeze(1).repeat(1, frame_length, 1, 1) # [batch_size, frame_length, agent_num, 256]

        hint = text_cond.reshape(batch_size, frame_length, agent_num, self.hint_feats)
        hint_a = hint[:,:,0]
        hint_b = hint[:,:,1] # (batch_size, frame_length, 26*3)
        # hint_a = self.first_zero_linear(hint_a)
        # hint_b = self.first_zero_linear(hint_b)
        # # TODO: if sequence_pos_encoder needed?
        # hint_a = self.sequence_pos_encoder(hint_a)
        # hint_b = self.sequence_pos_encoder(hint_b)

        h_a_prev, h_b_prev = h_prev[...,:self.latent_dim], h_prev[...,self.latent_dim:]

        mask = None
        if mask is not None:
            mask = mask[...,0]      
        if mask is None:
            mask = torch.ones(batch_size, frame_length).to(h_a_prev.device)
        key_padding_mask = ~(mask > 0.5)

        counterpart_mask = torch.ones(batch_size, frame_length, 1).to(h_a_prev.device)
        counterpart_mask[data['single_person']>0] = 0.
        
        h_a_prev = h_a_prev + hint_a
        h_b_prev = h_b_prev + hint_b

        control_a = []
        control_b = []
        for i, (block, zero_layer) in enumerate(zip(self.blocks, self.mid_zero_linear)):
                h_a = block(h_a_prev, h_b_prev * counterpart_mask, emb[:,:,0], key_padding_mask)
                h_b = block(h_b_prev, h_a_prev * counterpart_mask, emb[:,:,1], key_padding_mask)
                h_a_prev = h_a
                h_b_prev = h_b
                control_a.append(zero_layer(h_a))
                control_b.append(zero_layer(h_b))
                
        # control_a = torch.stack(control_a, dim=1) # (batch_size, num_layers, frame_length, latent_dim)
        # control_b = torch.stack(control_b, dim=1)
        # control = torch.cat([control_a, control_b], dim=-1) # (batch_size, num_layers, frame_length, latent_dim*2)
        return control_a, control_b
    
class cinterhuman_diffusion_phys(nn.Module):
    def __init__(self, base_model, smpl, **kwargs): 
        super(cinterhuman_diffusion_phys, self).__init__()       
        self.net = interhuman_diffusion_phys(smpl, **kwargs)
        state_dict = torch.load(base_model)['model']
        self.net.load_state_dict(state_dict, strict=False)
        self.net.requires_grad_(False)
        print("Loaded base model from", base_model)
        if not self.training:
            self.net.eval()
        
        self.control_net = ControlNet(**kwargs)
        self.control_net.requires_grad_(True)
        self.control_net.clip_transformer.requires_grad_(False)
        self.control_net.token_embedding.requires_grad_(False)
        self.control_net.ln_final.requires_grad_(False)

    def forward(self, data, **kwargs):

        batch_size, frame_length, agent_num = data['features'].shape[:3]
        num_valid = batch_size * frame_length * agent_num

        cond, img_info = self.net.condition_process(data)

        init_pose = data['init_pose_6d']
        noise, mean = self.net.generate_noise(init_pose)

        if self.training:
            x_start = self.net.input_process(data, img_info, mean)
            t, _ = self.net.sampler.sample(batch_size, x_start.device)
            # visualization
            viz_sampling = False
            if viz_sampling:
                self.visualize_sampling(x_start, t, data, img_info, mean, noise=noise)

            x_t = self.net.q_sample(x_start, t, noise=noise)
            pred = self.net.inference(x_t, t, cond, img_info, data, mean, control_net=self.control_net, **kwargs)

        else:
            if not self.net.eval_initialized:
                self.net.init_eval()
                self.net.eval_initialized = True
                
            pred = self.net.ddim_sample_loop(noise, mean, cond, img_info, data, control_net=self.control_net, **kwargs)
            
        return pred