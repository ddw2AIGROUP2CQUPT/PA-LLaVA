# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from pallava.flip.fflip import VisionModel,VisionConfig
from xtuner.registry import BUILDER
import torch.nn.functional as F
from xtuner.model.modules import dispatch_modules
from pallava.projector import (MLPProjectorConfig, MLPProjectorModel,
                                  QFormerProjectorConfig, QFormerProjectorModel)
from xtuner.model.modules.dispatch import SUPPORT_FLASH1, SUPPORT_FLASH2
from xtuner.model.utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    prepare_inputs_labels_for_multimodal, traverse_dict)


class LLaVAModel(BaseModel):
    def __init__(self,
                 llm,
                 visual_encoder,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 visual_select_layer=-2,
                 pretrained_pth=None,
                 projector_depth=1,
                 projector_type="mlp",
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 max_position_embeddings=None,
                 cross_attention_freq=1,
                 num_hidden_layers=6):
        super().__init__()
        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            if isinstance(llm, dict):
                llm = self._dispatch_lm_model_cfg(llm, max_position_embeddings)

            self.llm = self._build_from_cfg_or_module(llm)

            
            if isinstance(visual_encoder, dict):
                self.visual_encoder = self._build_from_cfg_or_module(visual_encoder)
            else:
                checkpoint = torch.load(os.path.join(visual_encoder, "plip_80w_196token.bin"),map_location="cpu")
                self.vision_config = VisionConfig().from_json_file(os.path.join(visual_encoder, "config.json"))
                self.visual_encoder = VisionModel.from_pretrained("openai/clip-vit-base-patch16", config = self.vision_config)
                self.visual_encoder.load_state_dict(checkpoint)


        self.llm.config.use_cache = False
        dispatch_modules(self.llm)
        

        self.projector_type = projector_type
        if projector_type == 'mlp':
            projector_config = MLPProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth)
            self.projector = MLPProjectorModel(projector_config).to(
                self.visual_encoder.dtype)
        elif projector_type == 'qformer_mlp':
            projector_config = QFormerProjectorConfig(
                visual_hidden_size=self.visual_encoder.config.hidden_size,
                llm_hidden_size=self.llm.config.hidden_size,
                depth=projector_depth,
                cross_attention_freq=cross_attention_freq,
                num_hidden_layers=num_hidden_layers)
            self.projector = QFormerProjectorModel(projector_config).to(
                self.visual_encoder.dtype)
        else:
            raise ValueError(
                f"Unsupported projector type: {projector_type}")
        print('Projector Parameters:', sum(p.numel() for p in self.projector.parameters()))

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)

            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.is_first_iter = True

        print('LLM trainable Parameters:', sum(p.numel() for p in self.llm.parameters() if p.requires_grad))


    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.visual_encoder.gradient_checkpointing_enable()
        self.projector.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.visual_encoder.gradient_checkpointing_disable()
        self.projector.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        return to_return

    @staticmethod
    def _prepare_for_long_context_training(cfg, llm_cfg,
                                           max_position_embeddings):

        orig_rope_scaling = getattr(llm_cfg, 'rope_scaling', None)
        if orig_rope_scaling is None:
            orig_rope_scaling = {'factor': 1}

        orig_rope_scaling_factor = orig_rope_scaling[
            'factor'] if 'factor' in orig_rope_scaling.keys() else 1
        orig_ctx_len = getattr(llm_cfg, 'max_position_embeddings', None)
        if orig_ctx_len:
            orig_ctx_len *= orig_rope_scaling_factor
            if max_position_embeddings > orig_ctx_len:
                scaling_factor = float(
                    math.ceil(max_position_embeddings / orig_ctx_len))
                llm_cfg.rope_scaling = {
                    'type': 'linear',
                    'factor': scaling_factor
                }

        # hardcode for internlm2
        llm_cfg.attn_implementation = 'flash_attention_2'
        cfg.config = llm_cfg

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_flash_attn(cfg, llm_cfg):
        cls_name = type(llm_cfg).__name__
        SUPPORT_SDPA_ATTN = ('LlamaConfig', 'GemmaConfig', 'MistralConfig',
                             'MixtralConfig', 'Qwen2Config', 'Qwen2MoeConfig',
                             'Starcoder2Config', 'Starcoder2Config',
                             'Phi3Config')
        SUPPORT_FLASH_ATTN2 = ('InternLM2Config', 'LlamaConfig', 'GemmaConfig',
                               'MistralConfig', 'MixtralConfig', 'Qwen2Config',
                               'Qwen2MoeConfig', 'Starcoder2Config',
                               'Starcoder2Config', 'Phi3Config')

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        if getattr(cfg, 'attn_implementation', None) is not None:
            # Flash Attention 2.0 only supports torch.float16 and
            # torch.bfloat16 dtypes
            if cfg.attn_implementation == 'flash_attention_2':
                cfg.torch_dtype = torch_dtype
        elif SUPPORT_FLASH2 and cls_name in SUPPORT_FLASH_ATTN2:
            cfg.torch_dtype = torch_dtype
            cfg.attn_implementation = 'flash_attention_2'
        elif SUPPORT_FLASH1 and cls_name in SUPPORT_SDPA_ATTN:
            cfg.attn_implementation = 'sdpa'

        return cfg, llm_cfg

    @staticmethod
    def _prepare_for_qlora_zero3(cfg):
        if (not is_deepspeed_zero3_enabled()) or (not hasattr(
                cfg, 'quantization_config')):
            return cfg

        torch_dtype = torch.bfloat16 if (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
            else torch.float16

        cfg.torch_dtype = torch_dtype
        quantization_config = cfg.quantization_config
        quantization_config.bnb_4bit_compute_dtype = torch_dtype
        quantization_config.bnb_4bit_quant_storage = torch_dtype

        return cfg

    def _dispatch_lm_model_cfg(self, cfg, max_position_embeddings=None):
        cfg = self._prepare_for_qlora_zero3(cfg)
        pretrained_model_name_or_path = cfg.pretrained_model_name_or_path
        llm_cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=True)
        cfg, llm_cfg = self._prepare_for_flash_attn(cfg, llm_cfg)
        if max_position_embeddings is not None:
            cfg, llm_cfg = self._prepare_for_long_context_training(
                cfg, llm_cfg, max_position_embeddings)
        return cfg

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):
        if self.is_first_iter:
            # hardcode for qlora DeepSpeed ZeRO3, put buffers and QuantState to
            # device
            # Only required in `LLaVAModel` .
            # We do not need this in `SupervisedFinetune` .
            self.to(data['input_ids'].device)
            self.is_first_iter = False
        if 'pixel_values' in data:
            if self.projector_type == 'qformer_mlp' and len(data['pixel_values']) > 0:
                # pad the pixel_values to max_patch_num
                bs = len(data['pixel_values'])
                max_patch_num = max(len(item) for item in data['pixel_values'])
                pad_patch_count = []
                for i in range(len(data['pixel_values'])):
                    current_patch_num = len(data['pixel_values'][i])
                    pad_patch_count.append(max_patch_num - current_patch_num)
                    
                    if current_patch_num < max_patch_num:
                        # nearest pad
                        pad_times = max_patch_num - current_patch_num
                        pad_tensor = data['pixel_values'][i][-1].unsqueeze(0).repeat(pad_times, 1, 1, 1)
                        padded_tensor = torch.cat([data['pixel_values'][i], pad_tensor], dim=0)
                        data['pixel_values'][i] = padded_tensor
                
                # data['pixel_values'].shape is [bs*num_patch, 3, 224, 224]
                data['pixel_values'] = torch.cat(data['pixel_values'], dim=0)
                visual_outputs = self.visual_encoder(
                    data['pixel_values'].to(self.visual_encoder.dtype),
                    output_hidden_states=True)
                
                seq_length, embed_dim = visual_outputs.hidden_states[self.visual_select_layer][:, 1:].shape[1:]
                visual_embeds = visual_outputs.hidden_states[self.visual_select_layer][:, 1:].view(bs, max_patch_num, seq_length, embed_dim)
                ori_pixel_embeds = visual_embeds[:, 0, :, :]
                patch_pixel_embeds = visual_embeds[:, 1:, :, :].reshape(bs, -1, embed_dim)

                patch_embeds_attns = torch.zeros(patch_pixel_embeds.size()[:-1], dtype=torch.long).to(ori_pixel_embeds.device)
                for i, count in enumerate(pad_patch_count):
                    actual_seq_length = (max_patch_num - count) * seq_length
                    patch_embeds_attns[i, :actual_seq_length] = 1
                pixel_values = self.projector(ori_pixel_embeds, patch_pixel_embeds, patch_embeds_attns)

            elif self.projector_type == 'mlp':
                visual_outputs  = self.visual_encoder(
                    torch.stack(data['pixel_values']).to(self.visual_encoder.dtype),
                    output_hidden_states=True)
                pixel_values = self.projector(
                    visual_outputs.hidden_states[self.visual_select_layer][:, 1:])
                
            else:
                raise NotImplementedError
            
            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal(llm=self.llm, **data)

        if mode == 'loss':
            return self.compute_loss(data, data_samples)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs
    
    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None):
        outputs = self.llm(**data)
        loss_dict = {'loss': outputs.loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)
