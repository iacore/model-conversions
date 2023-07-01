## Usage

```
# convert model from torch to safetensors (F32)
# weights must be F32 to be quantized
python ../torch-to-safetensors-f32.py ~/some-model.pth model-f32.safetensors

# plan
cargo run plan model-f32.safetensors plan.yaml

# edit plan
kak plan.yaml
# change 'keep' to 'q4_1' or other formats to quantize
# see `QuantizeTreatment` in src/main.rs for available formats

# quantize model
cargo run quantize model-f32.safetensors plan.yaml quantized.safetensors
```


The plan file should look like this after you edited it. It's YAML.

```
{
  metadata: null,
  catalog: {
    { dtype: F32, shape: [ 1, 1, 1024 ] }: {
      count: 120,
      treatment: keep,
    },
    { dtype: F32, shape: [ 1024, 4096 ] }: {
      count: 24,
      treatment: q4_2,
    },
    { dtype: F32, shape: [ 4096, 1024 ] }: {
      count: 24,
      treatment: q4_2,
    },
    { dtype: F32, shape: [ 1024, 1024 ] }: {
      count: 120,
      treatment: q4_2,
    },
    { dtype: F32, shape: [ 50277, 1024 ] }: {
      count: 2,
      treatment: keep,
    },
    { dtype: F32, shape: [ 1024 ] }: {
      count: 148,
      treatment: keep,
    },
  },
}
```
