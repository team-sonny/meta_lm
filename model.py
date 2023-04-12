import whisper
from torch import nn
from CustomModel import GPT2LMHeadModel
import torch
from encoder import TextEncoder, WavEncoder, PrefixEncoder
from config import Config
import torch

# 프롬프트 구조
# P-tunning + prompt, few shot in-context learning 으로 하면 좋을 것 같으니 비교 실험 필요.
# 질의 - 
prompt = ["다음 내용의 감정을 맞추시오. 예제: [기쁨, 놀람, 분노, 중립, 혐오, 공포, 슬픔]\n - 소리: "," - 텍스트: "," - 감정 값: "]

class MetaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = TextEncoder(Config.from_dict(eval(config.text_encoder)))
        self.wav_encoder = WavEncoder(Config.from_dict(eval(config.wav_encoder)))
        # GPI is Semi-Causal LM
        self.GPI = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2',output_hidden_states=True)
        self.prompt = config.prompt
        self.classifier = nn.Linear(768,7)
        self.dropout = nn.Dropout2d(config.dropout)
        self.GPI.transformer.wte = torch.nn.Identity()
        self.GPI.transformer.wpe = torch.nn.Identity()
        for param in self.GPI.parameters():
            param.requires_grad = False
            
        if self.prompt:
            self.prefix_encoder=PrefixEncoder(config)
            self.pre_seq_len = config.pre_seq_len
            self.n_layer = config.num_hidden_layers
            self.n_head = config.num_attention_heads
            self.n_embd = config.hidden_size // config.num_attention_heads
            self.prefix_tokens=torch.arange(self.pre_seq_len).long()
        
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1)
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
        
    def forward(self, inputs, order=False):
        """_summary_

        Args:
            order (bool, optional): first Audio, then Text if order is False. Defaults to False.

        Returns:
            _type_: _description_
        """
        batch_size = inputs['text_tokens']['input_ids'].shape[0]
        past_key_values = self.get_prompt(batch_size)
        text_tokens, wav_tokens = inputs['text_tokens'], inputs['wav_tokens']
        text_tokens = self.text_encoder(text_tokens)
        wav_tokens = self.wav_encoder(wav_tokens)
        inputs = torch.concat([text_tokens,wav_tokens],dim=1)
        # inputs = inputs.view(batch_size,-1,self.n_head,self.n_embd).permute([0,2,1,3])
        outputs = self.GPI(inputs,past_key_values=past_key_values)
        pred = self.classifier(outputs[2][11][-1])
        return pred, outputs
        
        