import whisper
from torch import nn
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
import torch
from encoder import TextEncoder, WavEncoder, PrefixEncoder

import torch


class MetaLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = TextEncoder(config)
        self.wav_encoder = WavEncoder(config)
        # GPI is Semi-Causal LM
        self.GPI = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.prompt = config.prompt
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        if self.prompt:
            self.prefix_encoder=PrefixEncoder(config)
            self.pre_seq_len = config.pre_seq_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            self.prefix_tokens=torch.arange(self.pre_seq_len).long()
        
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
        
    def forward(self,inputs, order=None):
        past_key_values = self.get_prompt(inputs.shape[0])
        text_tokens, wav_tokens = inputs['text_tokens'], inputs['wav_tokens']
        inputs = torch.concat([text_tokens,wav_tokens])
        return self.GPI(inputs,past_key_values=past_key_values)
        
        