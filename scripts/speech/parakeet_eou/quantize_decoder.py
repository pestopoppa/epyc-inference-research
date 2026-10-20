"""CS-46 (d): dynamic INT8 (QInt8 weights) copy of the self-exported fp32 decoder_joint.

The community int8 decoder (``decoder_joint-model.int8.onnx``) is measured as shipped; this makes the
same kind of graph from the step-wise export so the fp32-encoder + int8-decoder_joint pairing can also be
run through the exact step-wise decode path.  Encoder int8 is excluded (card: not exact).
usage: python -I quantize_decoder.py IN.onnx OUT.onnx
"""
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

quantize_dynamic(sys.argv[1], sys.argv[2], weight_type=QuantType.QInt8)
print("wrote", sys.argv[2])
