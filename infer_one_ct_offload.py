#!/usr/bin/env python3
"""
Memory-optimized single-volume Astra inference for a 15 GB NVIDIA T4.

The script preserves the merged Astra/Qwen FP16 weights and uses staged GPU
execution plus Accelerate CPU offload:

    1. Preprocess one CT on CPU.
    2. Run Merlin + Perceiver + projection on GPU.
    3. Delete the visual stack and release its GPU memory.
    4. Dispatch Qwen across GPU and CPU with a strict GPU-memory budget.
    5. Generate one report and save it under /root/capsule/results.

No command-line arguments are required.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Literal

# Set before importing torch/CUDA.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from accelerate import (
    dispatch_model,
    infer_auto_device_map,
    init_empty_weights,
)
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
from torch import einsum, nn
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


# ---------------------------------------------------------------------------
# Fixed Code Ocean paths and inference settings
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/root/capsule/code")
CT_PATH = Path(
    "/data/dataset/valid_preprocessed/valid_787/valid_787d/valid_787_d_1.nii.gz"
)
QWEN_PATH = Path("/root/capsule/data/qwen25_vl")
ASTRA_CHECKPOINT = Path(
    "/root/capsule/data/astra_weight/converted_weights.pth"
)
OUTPUT_PATH = Path("/root/capsule/results/generated_report.txt")
OFFLOAD_DIR = Path("/root/capsule/results/qwen_offload")

REGION: Literal["chest", "abdomen", "atlas"] = "chest"
CUDA_DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# Keep several GiB free for CUDA context, activations and KV cache.
# 11 GiB is intentionally conservative on a 14.58 GiB T4.
QWEN_GPU_BUDGET_GIB = 11

MIN_NEW_TOKENS = 2
MAX_NEW_TOKENS = 300
USE_CACHE = True

Region = Literal["chest", "abdomen", "atlas"]


# ---------------------------------------------------------------------------
# CT preprocessing
# ---------------------------------------------------------------------------
def _foreground_hu(x: torch.Tensor) -> torch.Tensor:
    return x > -1000


def build_preprocess(region: Region) -> Compose:
    """Match the repository's Merlin preprocessing."""
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
                    allow_smaller=False,
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
        raise ValueError("The demo accepts NIfTI only (.nii or .nii.gz).")

    item = build_preprocess(region)({"image": str(ct_path)})
    image = torch.as_tensor(item["image"], dtype=torch.float32)

    expected = (1, 224, 224, 160)
    if tuple(image.shape) != expected:
        raise RuntimeError(
            f"Unexpected preprocessed shape {tuple(image.shape)}; "
            f"expected {expected}."
        )

    return image.unsqueeze(0)  # [1, 1, 224, 224, 160]


# ---------------------------------------------------------------------------
# Perceiver modules copied from the Astra repository
# ---------------------------------------------------------------------------
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
    def __init__(
        self,
        *,
        dim: int,
        dim_head: int = 64,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads
        q = self.to_q(latents)
        k, v = self.to_kv(torch.cat((x, latents), dim=-2)).chunk(
            2,
            dim=-1,
        )

        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        similarity = einsum("... i d, ... j d -> ... i j", q, k)
        similarity = (
            similarity
            - similarity.amax(dim=-1, keepdim=True).detach()
        )
        attention = similarity.softmax(dim=-1)

        output = einsum("... i j, ... j d -> ... i d", attention, v)
        output = rearrange(
            output,
            "b h t n d -> b t n (h d)",
            h=h,
        )
        return self.to_out(output)


class PerceiverResampler(nn.Module):
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


# ---------------------------------------------------------------------------
# Astra model
# ---------------------------------------------------------------------------
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
    ) -> None:
        super().__init__()

        merlin_root = repo_root / "Merlin-main"
        if not merlin_root.exists():
            raise FileNotFoundError(
                f"Missing bundled Merlin code: {merlin_root}"
            )

        sys.path.insert(0, str(merlin_root))
        from merlin import Merlin

        # Merlin is initially created on CPU. Its checkpoint weights will be
        # overwritten by the merged Astra checkpoint.
        self.visual_encoder = Merlin(ImageEmbedding=True)

        self.qwen_processor = AutoProcessor.from_pretrained(
            str(qwen_path),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        self.tokenizer = self.qwen_processor.tokenizer

        # Create Qwen on the meta device rather than loading the base weights.
        # The merged Astra checkpoint already contains the full Qwen weights.
        config = AutoConfig.from_pretrained(
            str(qwen_path),
            trust_remote_code=True,
            local_files_only=True,
        )
        # Use PyTorch scaled-dot-product attention rather than the eager
        # attention path; this lowers attention-memory usage without requiring
        # the separately compiled flash-attn package.
        config._attn_implementation = "sdpa"

        with init_empty_weights(include_buffers=False):
            self.language_model = (
                Qwen2_5_VLForConditionalGeneration(config)
            )

        if hasattr(self.language_model, "visual"):
            del self.language_model.visual

        self.perceiver = PerceiverResampler(
            dim=2048,
            dim_head=256,
            heads=8,
            num_latents=32,
        )

        hidden_size = int(self.language_model.config.hidden_size)
        if hidden_size != 3584:
            raise ValueError(
                f"Checkpoint expects Qwen hidden size 3584, "
                f"got {hidden_size}."
            )

        self.llama_proj = nn.Linear(2048, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dtype = dtype
        self.language_device_map: dict[str, object] | None = None

    def load_astra_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Astra checkpoint not found: {checkpoint_path}"
            )

        print("Memory-mapping merged Astra checkpoint...")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise TypeError(
                "Unsupported checkpoint object; expected a state dictionary."
            )

        if state_dict and all(
            key.startswith("module.") for key in state_dict
        ):
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }

        incompatible = self.load_state_dict(
            state_dict,
            strict=False,
            assign=True,
        )

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        print(
            "Loaded merged Astra checkpoint. "
            f"Missing keys: {len(missing)}; "
            f"unexpected keys: {len(unexpected)}"
        )
        if missing:
            print("Missing keys:", missing[:20])
        if unexpected:
            print("Unexpected keys:", unexpected[:20])

        # The duplicate self.embed_tokens alias has deliberately been removed,
        # so no missing embedding alias should remain.
        if missing or unexpected:
            raise RuntimeError(
                "The merged checkpoint does not exactly match this model. "
                f"Missing={len(missing)}, unexpected={len(unexpected)}."
            )

        meta_parameters = [
            name
            for name, parameter in self.named_parameters()
            if parameter.is_meta
        ]
        if meta_parameters:
            raise RuntimeError(
                "Some parameters are still on the meta device: "
                f"{meta_parameters[:20]}"
            )

        del state_dict
        del checkpoint
        gc.collect()

        # assign=True preserves checkpoint tensor dtypes. The merged checkpoint
        # may have been saved in BF16, which a T4 should not use for this path.
        # Convert after releasing the checkpoint dictionary; conversion then
        # happens parameter-by-parameter instead of keeping two full state dicts.
        language_dtypes = {
            parameter.dtype
            for parameter in self.language_model.parameters()
            if parameter.is_floating_point()
        }
        print("Qwen checkpoint dtypes:", sorted(map(str, language_dtypes)))
        if language_dtypes != {self.dtype}:
            print(f"Converting Qwen weights to {self.dtype} on CPU...")
            self.language_model.to(dtype=self.dtype)
            gc.collect()

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

    @staticmethod
    def _cuda_memory(prefix: str) -> None:
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(
            f"{prefix}: CUDA allocated={allocated:.2f} GiB, "
            f"reserved={reserved:.2f} GiB"
        )

    @torch.inference_mode()
    def encode_image_then_release(
        self,
        image: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the visual stack first and then remove it from GPU memory."""
        print("Moving visual stack to the T4...")
        visual_modules = (
            self.visual_encoder,
            self.perceiver,
            self.llama_proj,
            self.layer_norm,
        )
        for module in visual_modules:
            module.eval()
            module.to(device=device, dtype=self.dtype)

        image = image.to(device=device, non_blocking=True)
        self._cuda_memory("Before visual encoding")

        with torch.autocast(
            device_type="cuda",
            dtype=self.dtype,
            enabled=True,
        ):
            image_features = self.visual_encoder(image)
            if image_features.ndim != 5:
                raise RuntimeError(
                    "Merlin must return a 5-D feature map, got "
                    f"{tuple(image_features.shape)}."
                )

            b, c, x, y, z = image_features.shape
            image_features = image_features.permute(0, 2, 3, 4, 1)
            image_features = image_features.reshape(
                b,
                x * y * z,
                c,
            )
            image_features = self.perceiver(
                image_features.unsqueeze(1).unsqueeze(1)
            ).squeeze(1)
            image_embeddings = self.llama_proj(image_features)
            image_embeddings = self.layer_norm(image_embeddings)

        # Keep only the tiny 32-token result; discard the CT and feature maps.
        image_embeddings = image_embeddings.detach()
        del image
        del image_features

        # The visual stack is not needed after producing image embeddings.
        del self.visual_encoder
        del self.perceiver
        del self.llama_proj
        del self.layer_norm

        gc.collect()
        torch.cuda.empty_cache()
        self._cuda_memory("After releasing visual stack")
        return image_embeddings

    def dispatch_qwen(
        self,
        *,
        gpu_budget_gib: int,
        offload_dir: Path,
    ) -> None:
        """Place most Qwen layers on GPU and overflow layers on CPU/disk."""
        offload_dir.mkdir(parents=True, exist_ok=True)

        max_memory = {
            0: f"{gpu_budget_gib}GiB",
            "cpu": _cpu_memory_budget(),
        }

        print("Computing Qwen device map with limits:", max_memory)
        device_map = infer_auto_device_map(
            self.language_model,
            max_memory=max_memory,
            no_split_module_classes=["Qwen2_5_VLDecoderLayer"],
            dtype=self.dtype,
            clean_result=False,
            offload_buffers=True,
            fallback_allocation=True,
            verbose=False,
        )

        # Keep token embedding on the main GPU so custom image embeddings and
        # text embeddings are assembled on the same device. Determine its exact
        # name dynamically because Transformers versions differ slightly.
        embedding_layer = self.language_model.get_input_embeddings()
        embedding_name = next(
            (
                name
                for name, module in self.language_model.named_modules()
                if module is embedding_layer
            ),
            None,
        )
        if embedding_name is None:
            raise RuntimeError("Could not locate Qwen input embedding module.")
        device_map[embedding_name] = 0
        print("Embedding module pinned to GPU:", embedding_name)

        print("Qwen device map summary:")
        counts: dict[str, int] = {}
        for target in device_map.values():
            key = str(target)
            counts[key] = counts.get(key, 0) + 1
        print(counts)

        self.language_model = dispatch_model(
            self.language_model,
            device_map=device_map,
            main_device=torch.device(CUDA_DEVICE),
            offload_dir=str(offload_dir),
            offload_buffers=True,
            force_hooks=True,
        )
        self.language_device_map = device_map
        self.language_model.eval()

        gc.collect()
        torch.cuda.empty_cache()
        self._cuda_memory("After Qwen dispatch")

    @torch.inference_mode()
    def generate_report(
        self,
        image_embeddings: torch.Tensor,
        *,
        region: Region,
        min_new_tokens: int,
        max_new_tokens: int,
        use_cache: bool,
    ) -> str:
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
            )
        finally:
            self.tokenizer.padding_side = old_padding_side

        # infer_auto_device_map is forced to keep embeddings on cuda:0.
        input_device = torch.device(CUDA_DEVICE)
        input_ids = tokens.input_ids.to(input_device)
        attention_mask = tokens.attention_mask.to(input_device)

        embedding_layer = self.language_model.get_input_embeddings()
        input_embeddings = embedding_layer(input_ids).clone()

        image_pad_id = self.tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )
        positions = (
            input_ids[0] == image_pad_id
        ).nonzero().flatten()

        if positions.numel() != 32:
            raise RuntimeError(
                "Prompt should contain 32 image tokens, found "
                f"{positions.numel()}."
            )

        image_embeddings = image_embeddings.to(
            device=input_embeddings.device,
            dtype=input_embeddings.dtype,
            non_blocking=True,
        )
        input_embeddings[0, positions] = image_embeddings[0]
        del image_embeddings
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()

        self._cuda_memory("Before report generation")

        generation_kwargs = {
            "inputs_embeds": input_embeddings,
            "attention_mask": attention_mask,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        try:
            generated = self.language_model.generate(
                **generation_kwargs
            )
        except torch.cuda.OutOfMemoryError:
            # Conservative fallback for unusually large allocator/cache peaks.
            print(
                "CUDA OOM with KV cache; retrying with use_cache=False "
                "and at most 64 generated tokens."
            )
            gc.collect()
            torch.cuda.empty_cache()
            generation_kwargs["use_cache"] = False
            generation_kwargs["max_new_tokens"] = min(
                max_new_tokens,
                64,
            )
            generated = self.language_model.generate(
                **generation_kwargs
            )

        text = self.qwen_processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()


# ---------------------------------------------------------------------------
# Resource and path helpers
# ---------------------------------------------------------------------------
def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cpu_memory_budget() -> str:
    """Return a conservative CPU-RAM budget for Accelerate."""
    available_bytes: int | None = None

    try:
        meminfo = Path("/proc/meminfo").read_text(
            encoding="utf-8"
        )
        values: dict[str, int] = {}
        for line in meminfo.splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            parts = raw.strip().split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0]) * 1024
        available_bytes = values.get("MemAvailable")
    except OSError:
        pass

    cgroup_root = Path("/sys/fs/cgroup")
    cgroup_max = _read_int(cgroup_root / "memory.max")
    cgroup_current = _read_int(cgroup_root / "memory.current")
    if cgroup_max is not None and cgroup_current is not None:
        cgroup_available = max(cgroup_max - cgroup_current, 0)
        available_bytes = (
            cgroup_available
            if available_bytes is None
            else min(available_bytes, cgroup_available)
        )

    # Leave at least 4 GiB for Python, MONAI, filesystem cache and hooks.
    if available_bytes is None:
        budget_gib = 12
    else:
        budget_gib = max(
            6,
            int(available_bytes / 1024**3) - 4,
        )

    return f"{budget_gib}GiB"


def validate_paths() -> None:
    required = {
        "repository": REPO_ROOT,
        "Merlin source": REPO_ROOT / "Merlin-main",
        "CT": CT_PATH,
        "Qwen model directory": QWEN_PATH,
        "Astra checkpoint": ASTRA_CHECKPOINT,
    }
    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required Code Ocean inputs:\n- "
            + "\n- ".join(missing)
        )


def main() -> None:
    validate_paths()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    gpu = torch.cuda.get_device_properties(0)
    print("GPU:", gpu.name)
    print(
        "GPU memory:",
        f"{gpu.total_memory / 1024**3:.2f} GiB",
    )
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CPU offload budget:", _cpu_memory_budget())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Preprocessing CT on CPU...")
    image = load_ct(CT_PATH, REGION)

    print("\n[2/5] Building Astra with empty Qwen parameters...")
    model = AstraInference(
        repo_root=REPO_ROOT,
        qwen_path=QWEN_PATH,
        dtype=DTYPE,
    )

    print("\n[3/5] Loading merged checkpoint...")
    model.load_astra_checkpoint(ASTRA_CHECKPOINT)
    model.eval()

    device = torch.device(CUDA_DEVICE)

    print("\n[4/5] Encoding CT and releasing the visual stack...")
    image_embeddings = model.encode_image_then_release(
        image,
        device,
    )
    del image
    gc.collect()

    print("\n[5/5] Dispatching Qwen and generating report...")
    model.dispatch_qwen(
        gpu_budget_gib=QWEN_GPU_BUDGET_GIB,
        offload_dir=OFFLOAD_DIR,
    )

    report = model.generate_report(
        image_embeddings,
        region=REGION,
        min_new_tokens=MIN_NEW_TOKENS,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=USE_CACHE,
    )

    print("\n===== Astra report =====\n")
    print(report)

    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()