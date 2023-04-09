import whisper
from torch import nn
from transformers import AutoModel
import torch

class PrefixEncoder(nn.Module):
    r'''
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.text_encoder = AutoModel.from_pretrained("Klue/RoBERTa-large")
        self.connector = nn.Linear(768,768)
        self.prompt=config.prompt
        
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
        if self.prompt:
            past_key_values = self.get_prompt(inputs.shape[0])
        else:
            past_key_values = None
        text_tokens = self.text_encoder(inputs,past_key_values=past_key_values)
        text_tokens = self.connector(text_tokens)
        return text_tokens
        
        
class WavEncoder(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.wav_encoder = whisper.load_model("base").encoder
        self.connector = nn.Linear(512,768)
        self.prompt=config.prompt
        
        for param in self.wav_encoder.parameters():
            param.requires_grad = False
        if self.prompt:
            self.prefix_encoder=PrefixEncoder(config)
            self.prefix_tokens
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
        if self.prompt:
            past_key_values = self.get_prompt(inputs.shape[0])
        else:
            past_key_values = None
        wav_tokens = self.wav_encoder(inputs,past_key_values=past_key_values,)
        wav_tokens = self.connector(wav_tokens)
        
        return wav_tokens
       
        