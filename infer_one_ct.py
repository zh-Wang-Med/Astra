#!/usr/bin/env python3
"""
Minimal single-volume inference demo for zh-Wang-Med/Astra.

Input:
    A raw-HU NIfTI CT volume (.nii or .nii.gz).
Output:
    One generated English radiology report.

This script intentionally avoids Astra's training DataModule, Lightning metrics,
and annotation JSON files. It reconstructs only the inference path used by
sft/models/Astra.py:
    Merlin -> 32-token Perceiver -> linear projection -> Qwen2.5-VL generate()
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

import torch
from einops import rearrange, repeat
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from peft import LoraConfig, TaskType, get_peft_model
from torch import einsum, nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


Region = Literal["chest", "abdomen", "atlas"]


def _foreground_hu(x: torch.Tensor) -> torch.Tensor:
    return x > -1000


def build_preprocess(region: Region) -> Compose:
    """Match the current repository's Merlin preprocessing."""
    if region == "chest":
        return Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                CropForegroundd(
                    keys=["image"],
                    source_key="image",
                    select_fn=_foreground_hu,
                ),
                Resized(keys=["image"], spatial_size=(224, 224, 160)),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1000,
                    a_max=200,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ToTensord(keys=["image"]),
            ]
        )

    # The repository uses the same image transform for its general abdomen
    # and Atlas-focused abdomen prompts.
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(
                keys=["image"],
                pixdim=(1.5, 1.5, 3.0),
                mode=("bilinear",),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=["image"], spatial_size=(224, 224, 160)),
            CenterSpatialCropd(keys=["image"], roi_size=(224, 224, 160)),
            ToTensord(keys=["image"]),
        ]
    )


def load_ct(ct_path: Path, region: Region) -> torch.Tensor:
    if not ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {ct_path}")
    name = ct_path.name.lower()
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise ValueError(
            "This minimal demo accepts NIfTI only (.nii or .nii.gz). "
            "Convert a DICOM series to NIfTI first."
        )

    item = build_preprocess(region)({"image": str(ct_path)})
    image = torch.as_tensor(item["image"], dtype=torch.float32)

    expected = (1, 224, 224, 160)
    if tuple(image.shape) != expected:
        raise RuntimeError(
            f"Unexpected preprocessed shape {tuple(image.shape)}; expected {expected}."
        )
    return image.unsqueeze(0)  # [B=1, C=1, 224, 224, 160]


def exists(value: object) -> bool:
    return value is not None


def feed_forward(dim: int, mult: int = 4) -> nn.Module:
    inner_dim = dim * mult
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    """Inference-only copy of sft/models/utils.py::PerceiverAttention."""

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(2, dim=-1)

        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        similarity = einsum("... i d, ... j d -> ... i j", q, k)
        similarity = similarity - similarity.amax(dim=-1, keepdim=True).detach()
        attention = similarity.softmax(dim=-1)

        output = einsum("... i j, ... j d -> ... i d", attention, v)
        output = rearrange(output, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(output)


class PerceiverResampler(nn.Module):
    """Inference-only copy of sft/models/utils.py::PerceiverResampler."""

    def __init__(
        self,
        *,
        dim: int,
        depth: int = 6,
        dim_head: int = 256,
        heads: int = 8,
        num_latents: int = 32,
        max_num_media: int | None = None,
        max_num_frames: int | None = None,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                        ),
                        feed_forward(dim=dim, mult=ff_mult),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F, V, D]
        b, media_count, frame_count, visual_count = x.shape[:4]

        if self.frame_embs is not None:
            frame_embs = repeat(
                self.frame_embs[:frame_count],
                "f d -> b t f v d",
                b=b,
                t=media_count,
                v=visual_count,
            )
            x = x + frame_embs

        x = rearrange(x, "b t f v d -> b t (f v) d")
        if self.media_time_embs is not None:
            x = x + self.media_time_embs[:media_count]

        latents = repeat(
            self.latents,
            "n d -> b t n d",
            b=b,
            t=media_count,
        )
        for attention, ff in self.layers:
            latents = attention(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class AstraInference(nn.Module):
    CHEST_PROMPT = (
        "Generate a comprehensive and detailed diagnosis report for this chest "
        "CT image. Structure the report by describing the following regions in "
        "this exact order: abdomen, bone, breast, esophagus, heart, lung, "
        "mediastinum, pleura, thyroid, trachea and bronchie. For any region "
        "without abnormalities, state 'normal.'."
    )
    ABDOMEN_PROMPT = (
        "Generate a comprehensive and detailed diagnosis report for this abdomen "
        "CT image. Structure the report by describing the following regions in "
        "this exact order: lower thorax, liver and biliary tree, gallbladder, "
        "spleen, pancreas, adrenal glands, kidneys and ureters, gastrointestinal "
        "tract, peritoneum, pelvic, vasculature, lymph nodes, musculoskeletal. "
        "For any region without abnormalities, state 'normal.'."
    )
    ATLAS_PROMPT = (
        "Please analyze the liver and biliary tree, pancreas, and kidneys and "
        "ureters areas from this abdominal CT scan. For any region without "
        "abnormalities, state 'normal.'."
    )

    def __init__(
        self,
        *,
        repo_root: Path,
        qwen_path: Path,
        dtype: torch.dtype,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        merlin_root = repo_root / "Merlin-main"
        if not merlin_root.exists():
            raise FileNotFoundError(
                f"Missing bundled Merlin code: {merlin_root}"
            )
        sys.path.insert(0, str(merlin_root))
        from merlin import Merlin  # imported after adding bundled package

        self.visual_encoder = Merlin(ImageEmbedding=True)

        self.qwen_processor = AutoProcessor.from_pretrained(
            str(qwen_path),
            trust_remote_code=True,
        )
        self.tokenizer = self.qwen_processor.tokenizer
        self.language_model = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(qwen_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        )

        if hasattr(self.language_model, "visual"):
            del self.language_model.visual

        # Keep this assignment before PEFT wrapping to match the training code's
        # module names and checkpoint keys.
        self.embed_tokens = self.language_model.get_input_embeddings()

        if use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.language_model = get_peft_model(
                self.language_model,
                peft_config,
            )

        self.perceiver = PerceiverResampler(
            dim=2048,
            dim_head=256,
            heads=8,
            num_latents=32,
        )
        hidden_size = int(self.language_model.config.hidden_size)
        if hidden_size != 3584:
            raise ValueError(
                f"Checkpoint expects Qwen hidden size 3584, got {hidden_size}. "
                "Use the same Qwen2.5-VL base model as training."
            )
        self.llama_proj = nn.Linear(2048, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def load_astra_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Astra checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(
                "Unsupported checkpoint object; expected a state-dict dictionary."
            )

        # Tolerate checkpoints saved from DistributedDataParallel.
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }

        incompatible = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded Astra checkpoint. Missing keys: "
            f"{len(incompatible.missing_keys)}; unexpected keys: "
            f"{len(incompatible.unexpected_keys)}"
        )
        if incompatible.missing_keys:
            print("First missing keys:", incompatible.missing_keys[:8])
        if incompatible.unexpected_keys:
            print("First unexpected keys:", incompatible.unexpected_keys[:8])

    def _prompt_for(self, region: Region) -> str:
        if region == "chest":
            instruction = self.CHEST_PROMPT
        elif region == "abdomen":
            instruction = self.ABDOMEN_PROMPT
        else:
            instruction = self.ATLAS_PROMPT

        image_tokens = "<|image_pad|>" * 32
        return (
            "<|im_start|>system\n"
            "You are a helpful assistant."
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"<|vision_start|>{image_tokens}<|vision_end|>"
            f"{instruction}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def encode_img(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.visual_encoder(image)
        if image_features.ndim != 5:
            raise RuntimeError(
                "Merlin must return a 5-D feature map, but got "
                f"shape {tuple(image_features.shape)}."
            )

        b, c, x, y, z = image_features.shape
        image_features = image_features.permute(0, 2, 3, 4, 1)
        image_features = image_features.reshape(b, x * y * z, c)

        image_features = self.perceiver(
            image_features.unsqueeze(1).unsqueeze(1)
        ).squeeze(1)
        image_features = self.llama_proj(image_features)
        return self.layer_norm(image_features)

    @torch.inference_mode()
    def generate_report(
        self,
        image: torch.Tensor,
        *,
        region: Region,
        min_new_tokens: int = 2,
        max_new_tokens: int = 600,
    ) -> str:
        image_embeddings = self.encode_img(image)
        prompt = self._prompt_for(region)

        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            tokens = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding="longest",
                truncation=False,
                add_special_tokens=False,
            ).to(image_embeddings.device)
        finally:
            self.tokenizer.padding_side = old_padding_side

        input_embeddings = self.embed_tokens(tokens.input_ids).clone()
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        positions = (tokens.input_ids[0] == image_pad_id).nonzero().flatten()
        if positions.numel() != 32:
            raise RuntimeError(
                f"Prompt should contain 32 image tokens, found "
                f"{positions.numel()}."
            )
        input_embeddings[0, positions] = image_embeddings[0]

        generated = self.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=tokens.attention_mask,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        text = self.qwen_processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one Astra report from one NIfTI CT volume."
    )
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument(
        "--region",
        choices=["chest", "abdomen", "atlas"],
        required=True,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Root of the cloned zh-Wang-Med/Astra repository.",
    )
    parser.add_argument(
        "--qwen-path",
        type=Path,
        required=True,
        help="Local Qwen2.5-VL base-model directory used by Astra.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Astra .pth checkpoint, e.g. base_e10.pth.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Use only for a checkpoint trained/saved without LoRA.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the default Astra 7B demo.")

    repo_root = args.repo_root.expanduser().resolve()
    qwen_path = args.qwen_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    ct_path = args.ct.expanduser().resolve()

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    print("Preprocessing CT...")
    image = load_ct(ct_path, args.region).to(device)

    print("Loading Astra...")
    model = AstraInference(
        repo_root=repo_root,
        qwen_path=qwen_path,
        dtype=dtype,
        use_lora=not args.no_lora,
    )
    model.load_astra_checkpoint(checkpoint)
    model.eval()
    model.to(device)

    autocast_enabled = device.type == "cuda" and dtype in {
        torch.float16,
        torch.bfloat16,
    }
    with torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=autocast_enabled,
    ):
        report = model.generate_report(
            image,
            region=args.region,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
        )

    print("\n===== Astra report =====\n")
    print(report)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
