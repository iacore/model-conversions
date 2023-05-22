Standalone safetensors reader and writer.

Not dependening on ggml, huggingface's impl or whatever.

Behaviour differences with https://github.com/huggingface/safetensors:

- Assumes the first 8 bytes (header len) is little endian
- Header JSON should be an object
    - every key started with `_` is ignored
- Doesn't care about data types, or alignments. You check it yourself. (maybe planned)
- Tensor data overlap is not checked. This only checks if the data is fully inside the file (no arbitrary memory read/write).
- Doesn't care about extra key-value pairs inside tensor info. As long as `dtype`, `shape`, `data_offsets` exist, it's good.
- strings JSON are only accepted when being valid utf-8
- no other "security" features
