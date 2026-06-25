# Export NVIDIA Nemotron Streaming ASR to ONNX
#
# Streaming latency is chosen at export time via --right-context (not at quantize
# or Rust load time). parakeet-rs mel chunk size must match the export:
#
#   Latency   --right-context   Mel frames/step   Samples/step @ 16 kHz
#   80 ms     0                 8                 1280
#   160 ms    1                 16                2560
#   560 ms    6 (default)       56                8960
#   1120 ms   13                112               17920
#
# Colab setup:
#   !pip install nemo_toolkit[asr] onnx onnxruntime soundfile
#
# Usage:
#   python export_nemotron_streaming.py model.nemo out_onnx/
#   python export_nemotron_streaming.py model.nemo out_onnx/ --right-context 13  # 1120 ms
#
# Output:
#   <output_dir>/
#     encoder.onnx          -- streaming encoder graph
#     encoder.onnx.data     -- encoder weights (external data, ~2.3 GB)
#     decoder_joint.onnx    -- RNNT decoder + joint network
#     tokenizer.model       -- SentencePiece tokenizer
#     config.json           -- streaming parameters, cache shapes, test I/O

import argparse
import functools
import gc
import glob
import json
import logging
import os
import shutil
import tarfile
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr

# NVIDIA-documented streaming modes (ms -> att_context right size).
# Mel frames per step = latency_ms / 10 (10 ms hop). Update CHUNK_SIZE in
# parakeet-rs/src/nemotron.rs to the same mel-frame count after changing this.
STREAMING_LATENCY_MS = {
    0: 80,
    1: 160,
    6: 560,
    13: 1120,
}

LATENCY_HELP = (
    "Streaming latency presets (--right-context -> ms): "
    + ", ".join(f"{rc}={ms}ms" for rc, ms in sorted(STREAMING_LATENCY_MS.items()))
    + ". Default 6 = 560 ms. Use 13 for 1120 ms."
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Export Nemotron streaming ASR model to ONNX",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Streaming latency is fixed at export time. Re-export and re-quantize "
        "to change it; then set CHUNK_SIZE in parakeet-rs/src/nemotron.rs to "
        "latency_ms / 10 mel frames.\n\n"
        "  ms     --right-context   mel frames   samples @ 16 kHz\n"
        "  80     0                 8            1280\n"
        "  160    1                 16           2560\n"
        "  560    6 (default)       56           8960\n"
        "  1120   13                112          17920\n"
    ),
)
parser.add_argument("input_path", help="Path to .nemo model file")
parser.add_argument("output_dir", help="Directory for ONNX outputs")
parser.add_argument(
    "--left-context", type=int, default=70,
    help="Attention left context in frames (default: 70; keep unless you know otherwise)",
)
parser.add_argument(
    "--right-context", type=int, default=6,
    help=LATENCY_HELP,
)
args = parser.parse_args()

ATT_CONTEXT_SIZE = [args.left_context, args.right_context]
OUTPUT_DIR = args.output_dir

# ---------------------------------------------------------------------------
# NeMo is extremely chatty, suppress most of it
# ---------------------------------------------------------------------------

logging.getLogger("nemo_logging").setLevel(logging.ERROR)
try:
    from nemo.core.classes.common import typecheck
    typecheck.set_typecheck_enabled(False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# PyTorch 2.9+ broke the old torch.onnx.export API by defaulting to dynamo.
# We patch it back to the legacy path, otherwise the export silently produces
# garbage or crashes. This took me a while to figure out on Colab...
# ---------------------------------------------------------------------------

pytorch_version = tuple(
    int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
)
print(f"PyTorch version: {torch.__version__}")

_PATCH_MARKER = "_legacy_onnx_patched"
if pytorch_version >= (2, 9) and not getattr(torch.onnx.export, _PATCH_MARKER, False):
    print("  Patching torch.onnx.export for PyTorch 2.9+ (dynamo=False)")
    _original_torch_onnx_export = torch.onnx.export

    @functools.wraps(_original_torch_onnx_export)
    def _patched_onnx_export(*pargs, **kwargs):
        if "dynamo" not in kwargs:
            kwargs["dynamo"] = False
        return _original_torch_onnx_export(*pargs, **kwargs)

    _patched_onnx_export._legacy_onnx_patched = True
    torch.onnx.export = _patched_onnx_export

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"\nLoading model from: {args.input_path}")

model = nemo_asr.models.ASRModel.restore_from(args.input_path, map_location=device)
model.eval()

print(f"  Model class : {type(model).__name__}")
print(f"  Encoder type: {type(model.encoder).__name__}")

# ---------------------------------------------------------------------------
# Extract tokenizer
# ---------------------------------------------------------------------------

print("\nExtracting tokenizer...")
with tarfile.open(args.input_path, "r:*") as tar:
    for member in tar.getnames():
        if member.endswith("tokenizer.model"):
            f = tar.extractfile(member)
            with open(os.path.join(OUTPUT_DIR, "tokenizer.model"), "wb") as out:
                out.write(f.read())
            print("  tokenizer.model extracted")
            break

# ---------------------------------------------------------------------------
# Configure streaming parameters
# ---------------------------------------------------------------------------

print("\nConfiguring streaming parameters...")

if hasattr(model.encoder, "set_default_att_context_size"):
    model.encoder.set_default_att_context_size(ATT_CONTEXT_SIZE)

streaming_cfg = model.encoder.streaming_cfg
print(f"  Streaming config: {streaming_cfg}")

subsampling_factor = model.cfg.encoder.get("subsampling_factor", 8)
left_context = ATT_CONTEXT_SIZE[0]
right_context = ATT_CONTEXT_SIZE[1]
chunk_size = right_context + 1  # output frames per chunk
latency_ms = (right_context + 1) * 80
if right_context in STREAMING_LATENCY_MS:
    print(f"  Streaming latency   : {STREAMING_LATENCY_MS[right_context]} ms (documented preset)")
else:
    print(f"  Streaming latency   : ~{latency_ms} ms (custom right-context)")

print(f"  Subsampling factor : {subsampling_factor}")
print(f"  Left context       : {left_context}")
print(f"  Right context      : {right_context}")
print(f"  Chunk size (frames): {chunk_size}")

model.encoder.setup_streaming_params(chunk_size=chunk_size, shift_size=chunk_size)

drop_extra_pre_encoded = getattr(streaming_cfg, "drop_extra_pre_encoded", 0)
print(f"  drop_extra_pre_encoded: {drop_extra_pre_encoded}")

# ---------------------------------------------------------------------------
# Initial cache state
# ---------------------------------------------------------------------------

batch_size = 1
cache_last_channel, cache_last_time, cache_last_channel_len = (
    model.encoder.get_initial_cache_state(batch_size=batch_size)
)

print(f"\nCache dimensions:")
print(f"  cache_last_channel    : {cache_last_channel.shape}")
print(f"  cache_last_time       : {cache_last_time.shape}")
print(f"  cache_last_channel_len: {cache_last_channel_len.shape}")

num_layers = cache_last_channel.shape[0]
hidden_dim = cache_last_channel.shape[3]
conv_context = cache_last_time.shape[3]

# ---------------------------------------------------------------------------
# Create test input via NeMo's streaming buffer.
# The buffer handles mel extraction and chunking internally we just feed
# it a dummy wav and grab the first chunk of mel features.
# ---------------------------------------------------------------------------

print("\nCreating test inputs via NeMo streaming buffer...")

from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

streaming_buffer = CacheAwareStreamingAudioBuffer(
    model=model,
    online_normalization=False,
    pad_and_drop_preencoded=True,
)

# Feed 2 seconds of noise -- just needs to be long enough for one full chunk
sample_rate = 16000
dummy_audio = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1

temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
sf.write(temp_wav.name, dummy_audio, sample_rate)
temp_wav.close()

streaming_buffer.append_audio_file(temp_wav.name, stream_id=-1)

# Buffer returns mel features directly
processed_signal, processed_signal_length = next(iter(streaming_buffer))
print(f"  Mel features shape : {processed_signal.shape}")
print(f"  Mel features length: {processed_signal_length}")

os.unlink(temp_wav.name)

# ---------------------------------------------------------------------------
# Run NeMo inference once so we can compare against the ONNX output later
# ---------------------------------------------------------------------------

print("\nRunning NeMo reference inference...")

with torch.no_grad():
    encoded, encoded_len, _, _, _ = model.encoder.cache_aware_stream_step(
        processed_signal=processed_signal,
        processed_signal_length=processed_signal_length,
        cache_last_channel=cache_last_channel,
        cache_last_time=cache_last_time,
        cache_last_channel_len=cache_last_channel_len,
        keep_all_outputs=False,
        drop_extra_pre_encoded=drop_extra_pre_encoded,
    )

print(f"  Encoder output shape : {encoded.shape}")
print(f"  Encoder output length: {encoded_len}")

# ---------------------------------------------------------------------------
# Export encoder
# ---------------------------------------------------------------------------

print("\nExporting encoder...")


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, drop_extra):
        super().__init__()
        self.encoder = encoder
        self.drop_extra = drop_extra

    def forward(
        self, processed_signal, processed_signal_length,
        cache_last_channel, cache_last_time, cache_last_channel_len,
    ):
        return self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )


encoder_wrapper = EncoderWrapper(model.encoder, drop_extra_pre_encoded)
encoder_wrapper.eval()

input_names = [
    "processed_signal",
    "processed_signal_length",
    "cache_last_channel",
    "cache_last_time",
    "cache_last_channel_len",
]
output_names = [
    "encoded",
    "encoded_len",
    "cache_last_channel_next",
    "cache_last_time_next",
    "cache_last_channel_len_next",
]

temp_encoder_path = os.path.join(OUTPUT_DIR, "encoder_temp.onnx")

torch.onnx.export(
    encoder_wrapper,
    (processed_signal, processed_signal_length,
     cache_last_channel, cache_last_time, cache_last_channel_len),
    temp_encoder_path,
    input_names=input_names,
    output_names=output_names,
    opset_version=17,
    dynamic_axes={
        "processed_signal": {0: "batch", 2: "time"},
        "processed_signal_length": {0: "batch"},
        "encoded": {0: "batch", 1: "time"},
        "encoded_len": {0: "batch"},
    },
)
print("  Encoder graph exported")

# PyTorch scatters weights across dozens of tiny files by default.
# Re-save everything into a single encoder.onnx + encoder.onnx.data pair
# so users only need to deal with two files.
print("  Consolidating encoder weights into single file...")

encoder_model = onnx.load(temp_encoder_path, load_external_data=True)
final_encoder_path = os.path.join(OUTPUT_DIR, "encoder.onnx")

onnx.save_model(
    encoder_model,
    final_encoder_path,
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="encoder.onnx.data",
    size_threshold=0,
)

del encoder_model
gc.collect()

# Clean up the scattered weight files from the initial export
for pattern in [
    "encoder_temp*", "*.weight", "*MatMul*",
    "Constant_*", "onnx__*", "encoder.pre_encode*",
]:
    for f in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
        try:
            os.remove(f)
        except OSError:
            pass

print("  Encoder saved: encoder.onnx + encoder.onnx.data")

# ---------------------------------------------------------------------------
# Export decoder/joint
# ---------------------------------------------------------------------------

print("\nExporting decoder/joint...")

temp_decoder_prefix = os.path.join(OUTPUT_DIR, "temp_model")
with torch.no_grad():
    model.export(output=temp_decoder_prefix + ".onnx", check_trace=False)

# NeMo's model.export() dumps both encoder and decoder -- we already have
# our own encoder export, so just grab the decoder_joint file and discard the rest
final_decoder_path = os.path.join(OUTPUT_DIR, "decoder_joint.onnx")
for f in glob.glob(os.path.join(OUTPUT_DIR, "*.onnx")):
    fname = os.path.basename(f).lower()
    if "decoder" in fname and "joint" in fname:
        if f != final_decoder_path:
            shutil.move(f, final_decoder_path)
        break

# Clean up temporary files
keep = {"encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model"}
for f in glob.glob(os.path.join(OUTPUT_DIR, "*")):
    if os.path.basename(f) not in keep and os.path.isfile(f):
        try:
            os.remove(f)
        except OSError:
            pass

print("  Decoder saved: decoder_joint.onnx")

# ---------------------------------------------------------------------------
# Save configuration
# ---------------------------------------------------------------------------

config = {
    "model_name": "nemotron-speech-streaming-en-0.6b",
    "sample_rate": 16000,
    "n_mels": 128,
    "subsampling_factor": subsampling_factor,
    "att_context_size": ATT_CONTEXT_SIZE,
    "left_context": left_context,
    "right_context": right_context,
    "chunk_size_output_frames": chunk_size,
    "drop_extra_pre_encoded": drop_extra_pre_encoded,
    "num_encoder_layers": num_layers,
    "hidden_dim": hidden_dim,
    "conv_context": conv_context,
    "vocab_size": model.tokenizer.vocab_size,
    "blank_id": model.tokenizer.vocab_size,
    "preprocessor": {
        "window_size": 0.025,
        "window_stride": 0.01,
        "n_fft": 512,
        "normalize": "per_feature",
        "preemph": 0.97,
    },
    "cache_shapes": {
        "cache_last_channel": list(cache_last_channel.shape),
        "cache_last_time": list(cache_last_time.shape),
        "cache_last_channel_len": list(cache_last_channel_len.shape),
    },
    "test_input": {
        "mel_shape": list(processed_signal.shape),
        "mel_length": int(processed_signal_length[0]),
    },
    "test_output": {
        "encoded_shape": list(encoded.shape),
        "encoded_len": int(encoded_len[0]),
    },
}

config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nConfiguration saved to {config_path}")

# ---------------------------------------------------------------------------
# Verify ONNX exports
# ---------------------------------------------------------------------------

print("\nVerifying ONNX exports...")

# -- Encoder
print("\n  Encoder:")
enc_session = ort.InferenceSession(final_encoder_path, providers=["CPUExecutionProvider"])

for inp in enc_session.get_inputs():
    print(f"    input  {inp.name}: {inp.shape}")
for out in enc_session.get_outputs():
    print(f"    output {out.name}: {out.shape}")

enc_outputs = enc_session.run(
    None,
    {
        "processed_signal": processed_signal.numpy(),
        "processed_signal_length": processed_signal_length.numpy(),
        "cache_last_channel": cache_last_channel.numpy(),
        "cache_last_time": cache_last_time.numpy(),
        "cache_last_channel_len": cache_last_channel_len.numpy(),
    },
)

nemo_out = encoded.numpy()
onnx_out = enc_outputs[0]
if nemo_out.shape == onnx_out.shape:
    diff = np.abs(nemo_out - onnx_out).max()
    print(f"    Output shape: {onnx_out.shape}, max diff vs NeMo: {diff:.6f}")
else:
    print(f"    Shape mismatch: NeMo={nemo_out.shape}, ONNX={onnx_out.shape}")

del enc_session

# -- Decoder
print("\n  Decoder:")
dec_session = ort.InferenceSession(final_decoder_path, providers=["CPUExecutionProvider"])

for inp in dec_session.get_inputs():
    print(f"    input  {inp.name}: {inp.shape}")
for out in dec_session.get_outputs():
    print(f"    output {out.name}: {out.shape}")

del dec_session
gc.collect()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Export complete")
print("=" * 60)

print(f"\nOutput directory: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 ** 2)
    print(f"  {f} ({size_mb:.1f} MB)")

print(f"\nTest: mel {list(processed_signal.shape)} -> encoded {list(encoded.shape)}")
