pub(crate) fn dtype_str(dtype: &safetensors::Dtype) -> &'static str {
    match dtype {
        safetensors::Dtype::BOOL => "BOOL",
        safetensors::Dtype::U8 => "U8",
        safetensors::Dtype::I8 => "I8",
        safetensors::Dtype::I16 => "I16",
        safetensors::Dtype::U16 => "U16",
        safetensors::Dtype::F16 => "F16",
        safetensors::Dtype::BF16 => "BF16",
        safetensors::Dtype::I32 => "I32",
        safetensors::Dtype::U32 => "U32",
        safetensors::Dtype::F32 => "F32",
        safetensors::Dtype::F64 => "F64",
        safetensors::Dtype::I64 => "I64",
        safetensors::Dtype::U64 => "U64",
        what => unreachable!("Unknown dtype: {what:?}"),
    }
}
