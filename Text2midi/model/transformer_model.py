# from aria.tokenizer import AbsTokenizer
# aria_tokenizer = AbsTokenizer()
import copy
import json
from typing import Optional, Any, Union, Callable
import torch.multiprocessing as mp
from torch.nn import DataParallel
import jsonlines
import math
import time
import torch
import os
import warnings
from tqdm import tqdm
from torch import Tensor
# from aria.tokenizer import AbsTokenizer
import pickle
from torch.nn import Module, LayerNorm, Dropout, Linear
from torch.nn.modules.container import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.multiprocessing import Process, set_start_method
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import torch.nn as nn

from st_moe_pytorch import MoE
from st_moe_pytorch import SparseMoEBlock

from einops import rearrange

from transformers import T5Tokenizer, T5EncoderModel

import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import torch.profiler

from accelerate import Accelerator
import torch
import os
from transformers import ClapModel, ClapProcessor
import tempfile
import numpy as np
import librosa
from midi2audio import FluidSynth
import argparse  # Add this import

class CaptionDataset(Dataset):
    def __init__(self, captions):
        self.captions = captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]

def custom_collate_fn(batch):
    captions = [item['caption'] for item in batch]
    locations = [item['location'] for item in batch]
    return captions, locations

def ensure_log_dir_exists(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer']

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    return cache.to(dtype=dtype)


@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    In-place RoPE. Credits to Katherine Crowson:
    x shape (b_sz, n_head, s_len, d_head).
    cos, sin shape (s_len, d_head // 2).
    """

    x = x.permute(0, 2, 1, 3)
    d = x.shape[-1] // 2
    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]
    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()
    # x1.mul_(cos).addcmul_(x2, sin, value=-1)
    # x2.mul_(cos).addcmul_(tmp, sin, value=1) ##was throwing some error: RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. You should replace the inplace operation by an out-of-place one.
    x1_new = x1.mul(cos) - x2.mul(sin)
    x2_new = x2.mul(cos) + tmp.mul(sin)
    x = torch.cat((x1_new, x2_new), dim=-1)
    x = x.permute(0, 2, 1, 3)
    
    return x


class MultiHeadSelfAttention(nn.Module):
    r"""Multi-head self-attention module.

    Args:
        embed_dim (int): The input embedding dimension.
        num_heads (int, optional): The number of attention heads (default: 4).
        dropout (float, optional): The dropout probability (default: 0.1).
        device (torch.device, optional): The device to use (default: None).
        dtype (torch.dtype, optional): The data type to use (default: None).

    Attributes:
        dim_head (int): The dimension of each attention head.
        scale (float): The scaling factor for attention scores.
        heads (int): The number of attention heads.
        to_qkv (nn.Linear): Linear layer for projecting input to query, key, and value.
        to_out (nn.Linear): Linear layer for projecting attention output to the original embedding dimension.
        dropout (nn.Dropout): Dropout layer.

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_first = batch_first
        self.dim_head = embed_dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.heads = num_heads
        hidden_dim = self.dim_head * num_heads
        self.to_qkv = nn.Linear(embed_dim, hidden_dim * 3, bias=False, **factory_kwargs)
        self.to_out = nn.Linear(hidden_dim, embed_dim, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        
        r"""Forward pass of the multi-head self-attention module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, sequence_length, embed_dim).

        """
        if not self.batch_first:
            x = x.transpose(0, 1)
        b, n, _ = x.size()
        q, k, v = torch.chunk(self.to_qkv(x), chunks=3, dim=-1)
        q, k, v = map(lambda t: t.contiguous().view(b, self.heads, n, -1), (q, k, v))

        self.freqs_cis = precompute_freqs_cis(
                seq_len=n,
                n_elem=self.embed_dim // self.heads,
                base=10000,
                dtype=x.dtype,
            ).to(x.device)
        freqs_cis = self.freqs_cis[: x.shape[1]]
        # q = apply_rotary_emb(q, freqs_cis)
        # k = apply_rotary_emb(k, freqs_cis)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.contiguous().view(b, n, -1)
        out = self.dropout(out)
        return self.to_out(out)


class Transformer(Module):
    r"""A transformer model.

    User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        use_moe: if True, use MoE instead of linear layer for feedforward network (default=False).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((32, 512))
        >>> tgt = torch.rand((32, 512, 30000))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, n_vocab: int = 30000, d_model: int = 512, nhead: int = 8, max_len: int = 5000,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, use_moe: bool = False, 
                 num_experts: int = 16, dropout: float = 0.1, 
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        self.use_moe = use_moe

        self.input_emb = nn.Embedding(n_vocab, d_model, **factory_kwargs)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len).to(device)

        # Load the FLAN-T5 encoder
        self.encoder = T5EncoderModel.from_pretrained("google/flan-t5-base").to(device)
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, use_moe, num_experts, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first,
                                                bias, **factory_kwargs)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, use_moe, decoder_norm)

        self.projection = nn.Linear(d_model, n_vocab).to(device)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self, src: Tensor, src_mask: Tensor, tgt: Tensor, memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: bool = True,
                memory_is_causal: bool = False) -> Tensor:
        r"""Take in and process masked source/target sequences.

        .. note::

            If a boolean tensor is provided for any of the [src/tgt/memory]_mask arguments, positions with a ``True`` value are
            not allowed to participate in the attention,
            which is the opposite of the definition for :attr:`attn_mask`
            in :func:`torch.nn.functional.scaled_dot_product_attention`.

        Args:
            src: the sequence to the encoder (required).
            src_attn_mask: the attention mask for the src sequence (required).
            tgt: the sequence to the decoder (required).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory_mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            - src: :math:`(S, S)` for unbatched input, :math:`(S, N)` if `batch_first=False` or
              `(N, S)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position :math:`i` is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where :math:`S` is the source sequence length, :math:`T` is the target sequence length, :math:`N` is the
            batch size, :math:`E` is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(src, tgt, src_mask=src_mask)
        """
        if src.dim() != tgt.dim():
            raise RuntimeError("the number of dimensions in src and tgt must be equal")

        memory = self.encoder(src, attention_mask=src_mask).last_hidden_state

        tgt = self.input_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        # tgt = tgt + tgt_pos
        
        if self.use_moe:
            with torch.cuda.amp.autocast(enabled =False):
                output, sum_total_aux_loss = self.decoder(tgt, memory, memory_mask=memory_mask,                                
                                    memory_key_padding_mask=memory_key_padding_mask,
                                    tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        else:
            output = self.decoder(tgt, memory, memory_mask=memory_mask,                                
                                memory_key_padding_mask=memory_key_padding_mask,
                                tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        
        output = self.projection(output)
        # output = F.log_softmax(output, dim=-1)

        if self.use_moe:
            return output, sum_total_aux_loss
        else:
            return output
        
    def generate(self, src: Tensor, src_mask: Tensor, r_tokenizer=None, caption='', tgt_fin = None, max_len: int = 100, temperature: float = 1.0):
        ## ADD A START OF SEQUENCE TOKEN  <SS> token to the src tensor
        r"""Generate a sequence of tokens from the given inputs.

        Args:
            src: the sequence to the encoder (required).
            src_mask: the attention mask for the src sequence (required).
            max_len: the maximum length of the sequence to generate (default=100).
            temperature: the temperature for the softmax (default=1.0).

        Returns:
            torch.Tensor: The generated sequence of tokens.

        """
        if src.dim() != 2:
            raise RuntimeError("The src tensor should be 2-dimensional")
        if tgt_fin is None:
            tgt_fin = torch.full((src.size(0), 1), 1, dtype=torch.long, device=src.device)
        else:
            tgt_fin = [0] + tgt_fin if isinstance(tgt_fin, list) else [0] + list(tgt_fin)
            value_tensor = torch.tensor(tgt_fin, dtype=torch.long, device=src.device)
            tgt_fin = value_tensor.unsqueeze(0).repeat(src.size(0), 1)
        # values = [21631, 8, 10, 9, 6, 7, 17, 21632, 11474, 20626, 21151, 9426, 20627, 21143, 11476, 20640, 21143, 11477, 20655, 21145, 11476, 20669, 21145, 11477, 20683, 21145, 13527, 20697, 21146, 13529, 20712, 21145, 7013, 20769, 21143, 7006, 20769, 21143, 7006, 20769, 21141, 7009, 20769, 21143, 9426, 20797, 21144, 11474, 20797, 21173, 11476, 20812, 21144, 11477, 20826, 21145, 11476, 20840, 21145, 11477, 20855, 21145, 13527, 20869, 21144, 13529, 20883, 21143, 7006, 20940, 21139, 7013, 20940, 21140, 7006, 20940, 21147, 7009, 20940, 21147, 11474, 20969, 21144, 11474, 20969, 21170, 11476, 20983, 21144, 11477, 20997, 21145, 11476, 21012, 21144, 11477, 21026, 21144, 11479, 21040]
        # values_tensor = torch.tensor(values, dtype=torch.long, device=src.device)
        # tgt_fin = values_tensor.unsqueeze(0).repeat(src.size(0), 1)
        for i in tqdm(range(max_len)):
            max_index = tgt_fin.max()
            # assert max_index < 21634, "tgt_fin contains index out of range. Adjust n_vocab or fix tgt_fin indices."
            tgt = tgt_fin
            if self.use_moe:
                output, _ = self.froward(src, src_mask, tgt, memory_mask=None,                                
                                memory_key_padding_mask=None,
                                tgt_is_causal=True, memory_is_causal=False)
            else:
                output = self.forward(src, src_mask, tgt, memory_mask=None,                                
                                      memory_key_padding_mask=None,
                                      tgt_is_causal=True, memory_is_causal=False)          
            # logits = self.projection(output)
            logits = output
            # Get logits for the last position
            last_token_logits = logits[:, -1, :]

            # Apply temperature and softmax to get probabilities
            probs = F.softmax(last_token_logits/temperature, dim=-1)

            # Get top 3 predictions and their probabilities
            top_probs, top_indices = torch.topk(probs, 3, dim=-1)

            # Print top 3 predictions
            # print("Top 3 predictions:")
            # for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
            #     token_id = idx.item()
            #     token_prob = prob.item()
            #     print(f"  {i+1}. Token ID: {token_id}, Probability: {token_prob:.4f}")
            
            
            output = F.log_softmax(logits/temperature, dim=-1)
            output = output.view(-1, output.size(-1))
            next_tokens = torch.multinomial(torch.exp(output), 1)[-1] # taking the last logit and adding to the sequence
            tgt_fin = torch.cat((tgt_fin, next_tokens.unsqueeze(-1)), dim=1)
            # Check if we should calculate CLAP score at this step
            # if i > 0 and i % 50 == 0:  # Check every 50 tokens
            #     try:
                    
            #         # Save current partial MIDI sequence
            #         temp_midi_path = tempfile.mktemp(suffix=".mid")
            #         temp_audio_path = tempfile.mktemp(suffix=".wav")
                    
            #         # Decode and save the current MIDI sequence
            #         partial_midi = r_tokenizer.decode(tgt_fin[0, 1:].tolist())
            #         partial_midi.dump_midi(temp_midi_path)
                    
            #         # Convert MIDI to audio using FluidSynth
            #         soundfont_path = "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # Default soundfont path
            #         if os.path.exists(soundfont_path):
            #             fs = FluidSynth(sound_font=soundfont_path)
            #             fs.midi_to_audio(temp_midi_path, temp_audio_path)
                        
            #             # Load audio
            #             try:
            #                 audio, sr = librosa.load(temp_audio_path, sr=48000, mono=True)
                            
            #                 # Load CLAP model
            #                 model_id = "laion/larger_clap_music"
            #                 clap_processor = ClapProcessor.from_pretrained(model_id)
            #                 clap_model = ClapModel.from_pretrained(model_id).to(src.device)
                            
            #                 # Prepare inputs
            #                 inputs = clap_processor(
            #                     text=[caption],
            #                     audios=list(audio),
            #                     sampling_rate=sr,
            #                     return_tensors="pt",
            #                     padding=True
            #                 ).to(src.device)
                            
            #                 # Calculate CLAP score
            #                 with torch.no_grad():
            #                     outputs = clap_model(**inputs)
            #                     logits_per_audio = outputs.logits_per_audio
            #                     clap_score = logits_per_audio[0].item()
                            
            #                 print(f"Step {i}: CLAP score = {clap_score:.4f}")
                            
            #                 # Clean up temporary files
            #                 os.remove(temp_midi_path)
            #                 os.remove(temp_audio_path)
            #             except Exception as e:
            #                 print(f"Error processing audio at step {i}: {e}")
            #         else:
            #             print(f"Soundfont not found at {soundfont_path}. Skipping CLAP score calculation.")
            #     except Exception as e:
            #         print(f"Error calculating CLAP score at step {i}: {e}")
        return tgt_fin[:, 1:]

    @staticmethod
    def generate_square_subsequent_mask(
            sz: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        r"""Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
        return _generate_square_subsequent_mask(sz, dtype=dtype, device=device)


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)




class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif encoder_layer.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn was passed bias=False"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False



    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        # is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # if not is_fastpath_enabled:
        #     why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output




class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    __constants__ = ['norm']

    def __init__(
        self,
        decoder_layer: "TransformerDecoderLayer",
        num_layers: int,
        use_moe: bool = False,
        norm: Optional[Module] = None
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_moe = use_moe
        self.norm = norm


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
        # print(f'target is causal: {tgt_is_causal}')

        if self.use_moe:
            sum_total_aux_loss = 0
            for mod in self.layers:
                output, total_aux_loss, balance_loss, router_z_loss = mod(output, memory,
                             memory_mask=memory_mask,
                             memory_key_padding_mask=memory_key_padding_mask,
                             tgt_is_causal=tgt_is_causal,
                             memory_is_causal=memory_is_causal)
                sum_total_aux_loss += total_aux_loss
        else:
            for mod in self.layers:
                output = mod(output, memory,
                            memory_mask=memory_mask,                            
                            memory_key_padding_mask=memory_key_padding_mask,
                            tgt_is_causal=tgt_is_causal,
                            memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        if self.use_moe:
            return output, sum_total_aux_loss
        else:
            return output



class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.

    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu



    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        # if not is_fastpath_enabled:
        #     why_not_sparsity_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif self.self_attn.in_proj_bias is None:
            why_not_sparsity_fast_path = "self_attn was passed bias=False"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)




class TransformerDecoderLayer(Module):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, use_moe: bool = False, num_experts: int = 16,
                 dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs) 
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        self.use_moe = use_moe

        if use_moe:
            self.moe = MoE(
                dim = d_model,
                num_experts = num_experts,      # increase the experts (# parameters) of your model without increasing computation
                gating_top_n = 2,               # default to top 2 gating, but can also be more (3 was tested in the paper with a lower threshold)
                threshold_train = 0.2,          # at what threshold to accept a token to be routed to second expert and beyond - 0.2 was optimal for 2 expert routing, and apparently should be lower for 3
                threshold_eval = 0.2,
                capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
                capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
                balance_loss_coef = 1e-2,       # multiplier on the auxiliary expert balancing auxiliary loss
                router_z_loss_coef = 1e-3,      # loss weight for router z-loss
            ).to(device)
            self.moe_block = SparseMoEBlock(
                self.moe,
                add_ff_before = True,
                add_ff_after = True
            ).to(device)
        else:
            # Implementation of Feedforward model
            self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
            self.dropout = Dropout(dropout)
            self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        # print(f'target is causal: {tgt_is_causal}')
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            if self.use_moe:
                m, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)
                x = x + m
            else:
                x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            if self.use_moe:
                m, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)
                x = x + m
            else:
                x = self.norm3(x + self._ff_block(x))

        if self.use_moe:
            return x, total_aux_loss, balance_loss, router_z_loss
        else:
            return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, is_causal=is_causal)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

def check_instruments(genereated_seq):
    ins_present = []
    ins_count = 0
    instrument_list = ["piano", "chromatic", "organ", "guitar", "bass", "strings", "ensemble", "brass", "reed", "drum", "pipe", "synth_lead", "synth_pad", "synth_effect", "ethnic", "percussive", "sfx"]
    for token in genereated_seq:
        try:
            ins, pitch, vel = token
            # print(str(ins))
        except ValueError:
            try:
                ins, pitch = token
            except ValueError:
                ins = token
        if str(ins) in instrument_list:
            # print('coming here')
            
            if ('prefix', 'instrument', str(ins)) not in ins_present and ins_count < 15:
                ins_count += 1
                print(f'adding instrument {ins}')
                ins_present.append(('prefix', 'instrument', str(ins)))
    if ins_present != []:
        genereated_seq = ins_present + ['<S>']+ genereated_seq +['<E>']
    else:
        genereated_seq = genereated_seq +['<E>']
    print(genereated_seq)
    return genereated_seq 

def process_caption(gpu_id, captions, model, tokenizer, r_tokenizer):
    # Detect device: CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        print(f"Using CUDA on GPU {gpu_id}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS on macOS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Move the model to the selected device
    model.to(device)
    model.eval()

    for caption in captions:
        src = caption['caption']
        location = caption['location']

        # Tokenize input
        inputs = tokenizer(src, return_tensors='pt', padding=True, truncation=True)
        input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
        input_ids = input_ids.to(device)
        attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
        attention_mask = attention_mask.to(device)

        # Generate output
        output = model.generate(input_ids, attention_mask, max_len=5000, temperature=0.9)
        output_list = output[0].tolist()

        # Decode MIDI and save it
        generated_midi = r_tokenizer.decode(output_list)
        generated_midi.dump_midi(f"../res/{location}")
        
def test_generate(caption):
    # Detect device: CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA on NVIDIA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS on macOS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    artifact_folder = '../artifacts'
    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    caption_dataset_path = '/root/text2midi/captions/train.json'
    print(f'caption_dataset_path: {caption_dataset_path}')

    # Load the tokenizer dictionary
    with open(tokenizer_filepath, "rb") as f:
        r_tokenizer = pickle.load(f)
    vocab_size = len(r_tokenizer)  # +1
    print("Vocab size: ", vocab_size)

    # Initialize model
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    model.load_state_dict(torch.load('/root/output_test_new/epoch_30/pytorch_model.bin', map_location=device))
    model.to(device)  # Move model to detected device
    model.eval()

    # Prepare input
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    # src = "A pop song with nostalgic feeling."
    # src = "A happy christmas song suitable for festive mood."
    # src = "A melodic electronic song with ambient elements, featuring piano, acoustic guitar, alto saxophone, string ensemble, and electric bass. Set in G minor with a 4/4 time signature, it moves at a lively Presto tempo. The composition evokes a blend of relaxation and darkness, with hints of happiness and a meditative quality."
    # src="An energetic and melodic electronic trance track with a space and retro vibe, featuring drums, distortion guitar, flute, synth bass, and slap bass. Set in A minor with a fast tempo of 138 BPM, the song maintains a 4/4 time signature throughout its duration."
    # src="A cheerful and melodic pop Christmas song featuring piano, acoustic guitar, vibraphone, bass, and drums, set in the key of Eb minor with a fast tempo of 123 bpm and a 4/4 time signature, creating a joyful and relaxing atmosphere."
    # src = "This short electronic song in C minor features a brass section, string ensemble, tenor saxophone, clean electric guitar, and slap bass, creating a melodic and slightly dark atmosphere. With a tempo of 124 BPM (Allegro) and a 4/4 time signature, the track incorporates a chord progression of C7/E, Eb6, and Bbm6, adding a touch of corporate and motivational vibes to the overall composition."
    # src="This motivational electronic and pop song features a clean electric guitar, rock organ, synth voice, acoustic guitar, and vibraphone, creating a melodic and uplifting atmosphere. Set in the key of G# minor with a 4/4 time signature, the track moves at an energetic Allegro tempo of 120 beats per minute. The chord progression of Bbm7 and F# adds to the song's inspiring and corporate feel."
    # src = "Played at 149 beats per minute in 2/4 time signature and the key of G major, classical piece with instruments: bassoon, clarinet, flute, horn, oboe, and trumpet."
    # src= 'Played at 114 beats per minute in 1/4 time signature and the key of g# minor, classical piece with the following instruments: clarinet, english horn, flute, horn, piccolo, trombone, and trumpet.'
    inputs = tokenizer(caption, return_tensors='pt', padding=True, truncation=True)
    input_ids = nn.utils.rnn.pad_sequence(inputs.input_ids, batch_first=True, padding_value=0)
    input_ids = input_ids.to(device)
    attention_mask = nn.utils.rnn.pad_sequence(inputs.attention_mask, batch_first=True, padding_value=0)
    attention_mask = attention_mask.to(device)
    output = model.generate(input_ids, attention_mask, r_tokenizer, caption, max_len=2000, temperature=0.9)
    output_list = output[0].tolist()

    # Decode and save MIDI
    generated_midi = r_tokenizer.decode(output_list)
    generated_midi.dump_midi(f"outputs/metal.mid")

def load_model_and_tokenizer(accelerator, model_path, vocab_size, tokenizer_filepath):
    device = accelerator.device
    with open(tokenizer_filepath, "rb") as f:
        r_tokenizer = pickle.load(f)
    model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    return model, tokenizer, r_tokenizer

def process_example(accelerator, model, tokenizer, r_tokenizer, example, location, output_path):
    device = accelerator.device
    inputs = tokenizer(example, return_tensors='pt', padding=True, truncation=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        output = model.module.generate(input_ids, attention_mask, r_tokenizer, max_len=2000, temperature=0.9)
    output_list = output[0].tolist()
    generated_midi = r_tokenizer.decode(output_list)
    generated_midi.dump_midi(output_path)

def run_accelerate_generation():
    accelerator = Accelerator()
    artifact_folder = '../artifacts'
    tokenizer_filepath = os.path.join(artifact_folder, "vocab_remi.pkl")
    model_path = '/root/output_test_new/epoch_30/pytorch_model.bin'
    captions_path = '/root/captions/train.json'
    
    with jsonlines.open(captions_path) as reader:
        selected_captions = [line for line in reader if line.get('test_set') is True]
    
    with open(tokenizer_filepath, "rb") as f:
        r_tokenizer = pickle.load(f)
    
    model, tokenizer, r_tokenizer = load_model_and_tokenizer(accelerator, model_path, len(r_tokenizer), tokenizer_filepath)
    model = accelerator.prepare(model)
    
    dataset = CaptionDataset(selected_captions)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False, collate_fn=custom_collate_fn)
    dataloader = accelerator.prepare(dataloader)
    
    for captions, locations in dataloader:
        for example, location in zip(captions, locations):
            output_path = os.path.join(f'/root/Text2midi/res_acc', location)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            process_example(accelerator, model, tokenizer, r_tokenizer, example, location, output_path)
            
# run_accelerate_generation() #uncomment this and comment __main__ to run accelerate generation

def main():
    parser = argparse.ArgumentParser(description="Generate MIDI from caption")
    parser.add_argument('--caption', type=str, required=True, help='Caption to generate MIDI from')
    args = parser.parse_args()
    test_generate(args.caption)

'''
comment out the next section function and uncomment the run_accelerate_generation() function to run the accelerate generation
'''
if __name__ == "__main__":
    main()
    print("Done")
