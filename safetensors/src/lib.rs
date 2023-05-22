use std::{collections::HashMap, io::Read};

use serde::Deserialize;
use serde_json::de::SliceRead;

pub struct SafetensorsHeader {
    /// the header data
    pub data: serde_json::Value,
    /// offset of end of header, from start of file
    pub offset_end_of_header: usize,
}

pub fn read_header(reader: &mut impl Read) -> Result<SafetensorsHeader, std::io::Error> {
    const EIGHT: usize = 8;
    let mut header_len = [0u8; EIGHT];
    reader.read_exact(&mut header_len)?;
    let header_len = u64::from_le_bytes(header_len);
    let mut header_raw = vec![0; header_len as usize];
    reader.read_exact(&mut header_raw)?;
    // the dance to make it ignore trailing data after JSON object
    let mut deserializer = serde_json::Deserializer::new(SliceRead::new(header_raw.as_slice()));
    let header = serde_json::Value::deserialize(&mut deserializer)?;
    Ok(SafetensorsHeader {
        data: header,
        offset_end_of_header: header_len as usize + EIGHT,
    })
}

pub struct TensorDataType {
    pub alignment: usize,
    /// size per unit datum
    /// for [`f16`] it's 2 (meaning 2 bytes)
    pub size: usize,
}

pub struct TensorInfo<'a> {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: &'a [u8],
}

pub enum WhereTheSliceStarts {
    /// start of slice is start of file
    StartOfFile,
    /// start of slice is right after header
    AfterHeader,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum LoadTensorError {
    #[error("invalid header: {why}")]
    InvalidHeader { why: String },
    #[error("duplicate tensor name: {0}")]
    DuplicateTensorName(String),
    #[error("offsets out of bounds: tensor={name:?} offsets={offsets:?}")]
    OffsetsOutOfBounds {
        name: String,
        offsets: (usize, usize),
    },
}

pub fn load_tensors<'a>(
    header: &SafetensorsHeader,
    slice: &'a [u8],
    slice_start_where: WhereTheSliceStarts,
) -> Result<HashMap<String, TensorInfo<'a>>, LoadTensorError> {
    let global_offset = match slice_start_where {
        WhereTheSliceStarts::StartOfFile => header.offset_end_of_header,
        WhereTheSliceStarts::AfterHeader => 0,
    };

    let Some(tensor_map) = header.data.as_object() else {
        return Err(LoadTensorError::InvalidHeader { why: "header is not JSON object".into() });
    };
    let mut result = HashMap::new();
    for (tensor_name, tensor_info_raw) in tensor_map {
        if tensor_name.starts_with("_") {
            continue;
        }
        let Some(tensor_info) = tensor_info_raw.as_object() else {
            return Err(LoadTensorError::InvalidHeader { why: format!("tensor info is not JSON object, tensor={tensor_name:?}") });
        };
        let dtype: String = tensor_info
            .get("dtype")
            .and_then(|v| Deserialize::deserialize(v).ok())
            .ok_or_else(|| LoadTensorError::InvalidHeader {
                why: format!(
                    "tensor info has no valid 'dtype' (should be string), tensor={tensor_name:?}"
                ),
            })?;

        let shape: Vec<usize> = tensor_info
            .get("shape")
            .and_then(|v| Deserialize::deserialize(v).ok())
            .ok_or_else(|| LoadTensorError::InvalidHeader {
                why: format!(
                    "tensor info has no valid 'shape' (should be list of numbers), tensor={tensor_name:?}"
                ),
            })?;
        let data_offsets: (usize, usize) = tensor_info
            .get("data_offsets")
            .and_then(|v| Deserialize::deserialize(v).ok())
            .ok_or_else(|| LoadTensorError::InvalidHeader {
                why: format!(
                    "tensor info has no valid 'data_offsets' (should be a pair of numbers), tensor={tensor_name:?}"
                ),
            })?;
        let Some(data) = slice.get(data_offsets.0 + global_offset..data_offsets.1 + global_offset) else {
            return Err(LoadTensorError::OffsetsOutOfBounds { name: tensor_name.clone(), offsets: data_offsets })
        };

        let None = result.insert(tensor_name.clone(), TensorInfo {
            dtype,
            shape,
            data,
        }) else {
            return Err(LoadTensorError::DuplicateTensorName(tensor_name.clone()));
        };
    }
    Ok(result)
}

#[test]
fn test_parse_trailing_data() {
    const SAMPLES: &[&[u8]] = &[b"\x05\0\0\0\0\0\0\0{}   ", b"\x05\0\0\0\0\0\0\0{}\0\0\0"];
    for sample in SAMPLES {
        let header = read_header(&mut sample.clone()).unwrap();
        _ = load_tensors(&header, &[], WhereTheSliceStarts::StartOfFile).unwrap();
    }
}

#[test]
fn test_sanity_check() {
    let sample = include_bytes!("../test_only_header.safetensors");
    let header = read_header(&mut sample.as_slice()).unwrap();
    let dummy_slice = unsafe { std::slice::from_raw_parts(0x070000000000 as *const u8, 966995048) };
    _ = load_tensors(&header, dummy_slice, WhereTheSliceStarts::StartOfFile).unwrap();
    _ = load_tensors(
        &header,
        &dummy_slice[55400..],
        WhereTheSliceStarts::AfterHeader,
    )
    .unwrap();
}
