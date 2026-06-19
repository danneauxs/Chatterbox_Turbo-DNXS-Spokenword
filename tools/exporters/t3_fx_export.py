#!/usr/bin/env python3
"""
Phase 1: FX trace + operator inventory for one T3 decoder block.

Outputs:
- Human-readable FX graph (stdout and optional file)
- JSON report with operator histogram and parameter summary

This does NOT use torch.onnx. It prepares for a manual ONNXScript export in Phase 2.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.fx as fx

import sys as _sys
# Ensure project modules are importable when run directly
_ROOT = Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_ROOT))
_sys.path.insert(0, str(_ROOT / "Chatterbox_API"))


def _find_ckpt_dir(cli_ckpt: str | None = None) -> Path | None:
    """Find the checkpoint directory based on CLI flag, environment variable, or configuration file.
    Args:
    cli_ckpt (str | None): Optional CLI flag specifying the checkpoint directory path.
    Returns:
    Path | None: The found checkpoint directory if it exists, otherwise None.
    """
    # 1) CLI flag
    if cli_ckpt:
        p = Path(cli_ckpt).expanduser()
        if p.exists():
            return p
    # 2) Env var
    ck = os.getenv("CHATTERBOX_CKPT_DIR", "").strip()
    if ck:
        p = Path(ck).expanduser()
        if p.exists():
            return p
    # 3) Config (best effort; optional)
    try:
        from config.config import CHATTERBOX_CKPT_DIR  # type: ignore
        ck = (CHATTERBOX_CKPT_DIR or "").strip()
        if ck and Path(ck).exists():
            return Path(ck)
    except Exception:
        pass
    # Try HF cache snapshots
    snap_root = Path.home() / ".cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots"
    if not snap_root.exists():
        return None
    # Pick latest snapshot that looks populated
    cands = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _load_t3(device: str = "cpu", ckpt_override: str | None = None):
    """Load a T3 model.
    Args:
    device (str): The device to load the model on, default is "cpu".
    ckpt_override (str | None): Path to a checkpoint file to override the default loading, optional.
    Returns:
    T3: The loaded T3 model instance.
    """
    # Patch rope_scaling for older transformers if needed
    try:
        import src.chatterbox.models.t3.llama_configs as llama_cfg
        cfg = llama_cfg.LLAMA_CONFIGS.get("Llama_520M")
        if isinstance(cfg, dict):
            rs = cfg.get("rope_scaling")
            if isinstance(rs, dict) and ("high_freq_factor" in rs or "rope_type" in rs):
                cfg["rope_scaling"] = {"type": "linear", "factor": float(rs.get("factor", 1.0))}
    except Exception:
        pass

    from safetensors.torch import load_file
    from src.chatterbox.models.t3 import T3

    ckpt_dir = _find_ckpt_dir(ckpt_override)
    if ckpt_dir is None:
        raise FileNotFoundError("Could not locate checkpoint dir. Use --ckpt or set CHATTERBOX_CKPT_DIR.")

    # Find a safetensors file
    chosen = None
    for root, _, files in os.walk(ckpt_dir):
        root_p = Path(root)
        if len(root_p.relative_to(ckpt_dir).parts) > 3:
            continue
        for f in files:
            if f.endswith(".safetensors") and ("t3" in f.lower() or f.lower().endswith(".safetensors")):
                chosen = root_p / f
                break
        if chosen is not None:
            break
    if chosen is None:
        raise FileNotFoundError(f"No .safetensors found under {ckpt_dir}")

    print(f"[fx] Using weights: {chosen}")
    t3 = T3()
    state = load_file(str(chosen))
    # Handle nested formats
    if isinstance(state, dict) and "model" in state:
        v = state["model"]
        if isinstance(v, dict):
            state = v
        elif isinstance(v, (list, tuple)) and v:
            state = v[0]
    missing, unexpected = t3.load_state_dict(state, strict=False)
    if missing:
        print(f"[fx] Missing keys: {len(missing)} (tolerated)")
    if unexpected:
        print(f"[fx] Unexpected keys: {len(unexpected)} (tolerated)")
    return t3.to(device).eval()


def _get_llama_layers(t3) -> List[torch.nn.Module]:
    """Retrieves the layers of a transformer model.
    Args:
    t3: The transformer object containing the layers.
    Returns:
    A list of PyTorch modules representing the layers of the transformer.
    """
    # Try common access patterns across transformers versions
    tfmr = t3.tfmr
    for attr in ("layers", "model", "decoder"):
        if hasattr(tfmr, attr):
            obj = getattr(tfmr, attr)
            if isinstance(obj, torch.nn.ModuleList):
                return list(obj)
            if hasattr(obj, "layers") and isinstance(obj.layers, torch.nn.ModuleList):
                return list(obj.layers)
    # Fallback: scan children for list of repeated blocks
    blocks = []
    for m in tfmr.modules():
        if type(m).__name__.lower().endswith("layer") and len(list(m.children())) > 0:
            blocks.append(m)
    if blocks:
        return blocks
    raise RuntimeError("Could not locate Llama decoder layers on t3.tfmr")


class BlockWrapper(torch.nn.Module):
    """Wrapper for a block to handle position IDs and suppress unnecessary outputs during tracing."""
    def __init__(self, block: torch.nn.Module):
        """Initialize a module wrapper for forward pass.
        Args:
        block (torch.nn.Module): The neural network module to wrap.
        hidden_states (torch.Tensor): Input tensor of hidden states.
        position_ids (torch.Tensor): Tensor containing positional information.
        attn_mask (torch.Tensor | None, optional): Attention mask; defaults to None.
        Returns:
        The output of the wrapped block's forward method.
        """
        super().__init__()
        self.block = block

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """Performs a forward pass through the model block using the provided hidden states and position IDs, optionally applying an attention mask.
        Args:
        hidden_states (torch.Tensor): Input tensor representing the hidden states.
        position_ids (torch.Tensor): Tensor containing position indices.
        attn_mask (torch.Tensor | None, optional): Attention mask to apply. Defaults to None.
        Returns:
        torch.Tensor: Output tensor after processing through the block.
        """
        # Pass through position_ids to avoid constructing tensors inside trace
        try:
            out = self.block(
                hidden_states,
                attention_mask=attn_mask,
                position_ids=position_ids,
                output_attentions=False,
                use_cache=False,
            )
        except TypeError:
            try:
                out = self.block(hidden_states, position_ids=position_ids)
            except TypeError:
                out = self.block(hidden_states)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


class RoPETracer(fx.Tracer):
    """Class for tracing a neural network block with rotary positional encoding using FX tracer.
    Extends fx.Tracer to identify specific modules during tracing.
    """
    def is_leaf_module(self, m: torch.nn.Module, qualname: str) -> bool:  # type: ignore[override]
        name = type(m).__name__
        if name in {"LlamaRotaryEmbedding", "RotaryEmbedding", "LlamaRMSNorm"}:
            return True
        return super().is_leaf_module(m, qualname)


def trace_block(block: torch.nn.Module, seq_len: int = 32, dim: int = 1024, device: str = "cpu") -> fx.GraphModule:
    """Returns a histogram of operation types in the given FX GraphModule.
    Args:
    gm (fx.GraphModule): The FX GraphModule to analyze.
    """
    ex = torch.randn(1, seq_len, dim, device=device)
    pos = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(1, -1)
    attn_mask = None
    wrapper = BlockWrapper(block).to(device).eval()
    # Warm-up with concrete tensors (not part of trace)
    with torch.no_grad():
        _ = wrapper(ex, pos, attn_mask)
    # Use custom tracer to keep rotary embedding as a leaf (avoid dtype Proxy ops)
    tracer = RoPETracer()
    gm = fx.GraphModule(wrapper, tracer.trace(wrapper, concrete_args={"position_ids": pos, "attn_mask": None}))
    return gm


def op_histogram(gm: fx.GraphModule) -> Dict[str, int]:
    """Generates a histogram of operation types in a given FX GraphModule.
    Args:
    gm (fx.GraphModule): The FX GraphModule to analyze.
    Returns:
    Dict[str, int]: A dictionary mapping operation types to their counts, sorted by count and then alphabetically.
    """
    hist: Dict[str, int] = {}
    for n in gm.graph.nodes:
        k = None
        if n.op == "call_function":
            k = getattr(n.target, "__name__", str(n.target))
        elif n.op == "call_method":
            k = f"method::{n.target}"
        elif n.op == "call_module":
            mod = gm.get_submodule(n.target)
            k = f"module::{type(mod).__name__}"
        else:
            k = n.op
        hist[k] = hist.get(k, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))


def main() -> int:
    """Parse command-line arguments for tracing a T3 decoder block and generate a report.
    Args:
    --layer (int): Decoder block index to trace.
    --seq-len (int): Sequence length for example input.
    --hidden-dim (int): Hidden size for example input.
    --device (str): Tracing device ("cpu" or "cuda").
    --report (str): Path to write JSON report.
    --graph-txt (str): Path to write FX graph text.
    --ckpt (str): Path to checkpoint directory.
    Returns:
    int: Exit status.
    """
    ap = argparse.ArgumentParser(description="FX trace a T3 decoder block and report ops")
    ap.add_argument("--layer", type=int, default=0, help="Decoder block index to trace")
    ap.add_argument("--seq-len", type=int, default=32, help="Sequence length for example input")
    ap.add_argument("--hidden-dim", type=int, default=1024, help="Hidden size for example input")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Tracing device")
    ap.add_argument("--report", default="out/t3_block_fx_report.json", help="Path to write JSON report")
    ap.add_argument("--graph-txt", default="out/t3_block_fx_graph.txt", help="Path to write FX graph text")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint directory (contains *.safetensors)")
    args = ap.parse_args()

    os.makedirs(Path(args.report).parent, exist_ok=True)

    t3 = _load_t3(device=args.device, ckpt_override=args.ckpt)
    layers = _get_llama_layers(t3)
    if not (0 <= args.layer < len(layers)):
        raise IndexError(f"Layer index {args.layer} out of range (0..{len(layers)-1})")
    block = layers[args.layer]
    print(f"[fx] Tracing decoder block {args.layer} ({type(block).__name__})")

    # Prefer the model's hidden size for tracing if not explicitly set
    hidden_dim = args.hidden_dim or getattr(t3, "dim", args.hidden_dim)
    gm = trace_block(block, seq_len=args.seq_len, dim=hidden_dim, device=args.device)
    graph_str = str(gm.graph)
    print("\n==== FX Graph (truncated) ====")
    print("\n".join(graph_str.splitlines()[:120]))

    # Operator histogram
    hist = op_histogram(gm)
    print("\n==== Operator histogram ====")
    for k, v in hist.items():
        print(f"{k:32s} : {v}")

    # Parameter summary for the block
    params = sum(p.numel() for p in block.parameters())
    trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)

    # Write artifacts
    Path(args.graph_txt).write_text(graph_str)
    report = {
        "layer_index": args.layer,
        "layer_type": type(block).__name__,
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "device": args.device,
        "param_count": params,
        "param_trainable": trainable,
        "op_histogram": hist,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"\n[fx] Wrote report to {args.report}")
    print(f"[fx] Wrote graph to  {args.graph_txt}")
    print("[fx] Phase 1 complete. Ready to implement ONNXScript mapping for listed ops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
