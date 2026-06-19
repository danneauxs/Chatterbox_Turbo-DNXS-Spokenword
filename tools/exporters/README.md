T3 Manual ONNX Exporter (FX -> ONNXScript)

Goal
- Produce a full, correct ONNX for the T3 forward path by tracing and translating ops, bypassing torch.onnx.

Phase 1 (this tool)
- Loads your T3 weights
- Isolates a single transformer decoder block from the HF Llama backbone
- FX-traces the block on representative inputs
- Emits a report of the FX graph and the ATen ops found (to guide op mapping)
- Scaffolds an ONNXScript translator and verifies environment

Usage
- Activate your venv
- Run: `python tools/exporters/t3_fx_export.py --layer 0 --report out/t3_block0_fx.json`
  - Loads weights from `config.CHATTERBOX_CKPT_DIR` or auto-discovers under `~/.cache/huggingface/hub/.../snapshots`
  - Writes: FX graph text, operator histogram, parameter summary

Next phases
- Implement op translators for Linear/LayerNorm/MatMul/Softmax/Reshape/Permute/Embedding/RoPE
- Assemble a full ONNX via ONNXScript and validate against PyTorch per‑layer

