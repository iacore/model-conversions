What's included in this repo:

- `model-conversion-wizard`
    - conversion from pytorch file to safetensors, without loading all in memory
    - interactive quantization from and to safetensors
- `safetensors`, a Rust crate that handles reading safetensors file. Not used yet.
- some python scripts

## Usage

```shell
cd conversion-wizard
cargo run # this will print usage
```

## References

See [safetensors](https://github.com/huggingface/safetensors) repo for details about safetensors.

See [rwkv.cpp](https://github.com/saharNooby/rwkv.cpp/tree/master/rwkv) for conversion form pth to rwkv.cpp-specific ggml format.

See [llama.cpp](https://github.com/ggerganov/llama.cpp) for conversion from pth to llama.cpp-specific ggml format. (The *.py files)

