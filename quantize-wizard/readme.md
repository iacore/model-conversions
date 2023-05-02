## Usage

```
# convert model to F32
# weights must be F32 to be quantized
python ../torch-to-safetensors-f32.py ~/some-model.pth model-f32.saft

# plan
cargo run plan model-f32.saft plan.yaml

# edit plan
kak plan.yaml
# change 'keep' to 'q4_1' or other formats to quantize
# see `QuantizeTreatment` in src/main.rs for available formats

# convert
cargo run convert model-f32.saft plan.yaml quantized.saft
```
