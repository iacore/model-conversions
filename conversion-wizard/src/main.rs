mod util;

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::{borrow::Cow, io::Write};

use anyhow::{bail, ensure, Context};
use clap::Parser;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Safetensors Interactive Quantization Tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
enum Args {
    ConvertPyTorch(ConvertPyTorchArgs),
    Plan(PlanArgs),
    Quantize(QuantizeArgs),
}

#[derive(Parser, Debug)]
struct ConvertPyTorchArgs {
    /// input pytorch (.pth) file
    #[arg()]
    model_in: PathBuf,

    /// output .safetensors file
    #[arg()]
    model_out: PathBuf,
}

#[derive(Parser, Debug)]
struct PlanArgs {
    /// input .safetensors file
    #[arg()]
    model_in: PathBuf,

    /// output plan (YAML) file name
    #[arg()]
    plan_out: PathBuf,
}

#[derive(Parser, Debug)]
struct QuantizeArgs {
    /// input .safetensors file
    #[arg()]
    model_in: PathBuf,

    /// plan file
    #[arg()]
    plan: String,

    /// output .safetensors file name
    #[arg()]
    model_out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args {
        Args::ConvertPyTorch(args) => do_the_pickle_thing(args),
        Args::Plan(args) => do_plan(args),
        Args::Quantize(args) => do_quantize(args),
    }
}

/// A view of a Tensor within the file.
/// Contains references to data within the full byte-buffer
/// And is thus a readable view of a single tensor
#[derive(Debug, PartialEq, Eq)]
pub struct TensorView<'data> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'data [u8],
}

impl<'data> safetensors::View for TensorView<'data> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<[u8]> {
        self.data.into()
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// load pytorch pickle file gracefully
fn do_the_pickle_thing(args: ConvertPyTorchArgs) -> Result<(), anyhow::Error> {
    let file = File::open(&args.model_in)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = repugnant_pickle::torch::RepugnantTorchTensors::new_from_file(&args.model_in)?;
    let mut tensor_slices: Vec<(String, TensorView)> = vec![];
    for tensor in tensors {
        tensor_slices.push((
            tensor.name,
            TensorView {
                dtype: match tensor.tensor_type {
                    repugnant_pickle::TensorType::Float64 => Dtype::F64,
                    repugnant_pickle::TensorType::Float32 => Dtype::F32,
                    repugnant_pickle::TensorType::Float16 => Dtype::F16,
                    repugnant_pickle::TensorType::BFloat16 => Dtype::BF16,
                    repugnant_pickle::TensorType::Int64 => Dtype::I64,
                    repugnant_pickle::TensorType::Int32 => Dtype::I32,
                    repugnant_pickle::TensorType::Int16 => Dtype::I16,
                    repugnant_pickle::TensorType::Int8 => Dtype::I8,
                    repugnant_pickle::TensorType::UInt8 => Dtype::U8,
                    repugnant_pickle::TensorType::Unknown(unknown_datatype) => {
                        bail!("unknown tensor datatype: {unknown_datatype}")
                    }
                },
                shape: tensor.shape.clone(),
                data: {
                    let start_offset = tensor.absolute_offset as usize;
                    let len: usize = tensor.shape.into_iter().product();
                    let end_offset = start_offset + len * tensor.tensor_type.size();
                    &mmap[start_offset..end_offset]
                },
            },
        ));
    }
    safetensors::serialize_to_file(tensor_slices, &None, &args.model_out)?;
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct QuantizePlan {
    metadata: Option<HashMap<String, String>>,
    catalog: HashMap<QuantizeKey, QuantizeInfo>,
}

#[derive(Hash, PartialEq, Eq, Serialize, Deserialize, Debug, Clone)]
struct QuantizeKey {
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Default, Serialize, Deserialize)]
struct QuantizeInfo {
    /// what tensors are of this shape
    pub count: usize,
    pub treatment: QuantizeTreatment,
}

#[allow(non_camel_case_types)]
#[derive(Default, Serialize, Deserialize)]
enum QuantizeTreatment {
    #[default]
    keep,
    q4_0,
    q4_1,
    // q4_2,
    q5_0,
    q5_1,
    q8_0,
}

fn do_plan(args: PlanArgs) -> Result<(), anyhow::Error> {
    let file = File::open(args.model_in)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let (_, header) = SafeTensors::read_metadata(&buffer)?;
    let metadata = header.metadata().clone();

    let mut tensors: HashMap<QuantizeKey, QuantizeInfo> = HashMap::new();
    for (_tensor_name, tensor_info) in header.tensors() {
        let k = QuantizeKey {
            dtype: util::dtype_str(&tensor_info.dtype).to_owned(),
            shape: tensor_info.shape.clone(),
        };
        let info = match tensors.get_mut(&k) {
            Some(v) => v,
            None => {
                assert!(tensors.insert(k.clone(), Default::default()).is_none());
                tensors.get_mut(&k).unwrap()
            }
        };

        info.count += 1;
    }

    let writer = File::create(args.plan_out)?;
    cyrly::to_writer(
        &writer,
        &QuantizePlan {
            metadata,
            catalog: tensors,
        },
    )?;

    Ok(())
}

fn do_quantize(args: QuantizeArgs) -> anyhow::Result<()> {
    let plan_file = File::open(args.plan)?;
    let plan: QuantizePlan = serde_yaml::from_reader(plan_file)?;
    let file = File::open(args.model_in)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut out = serde_json::Map::new();
    if let Some(metadata) = plan.metadata {
        out.insert("__metadata__".into(), serde_json::to_value(metadata)?);
    }

    let mut write_plan = TensorPlaner::default();

    for (name, tensor) in tensors.tensors() {
        let dtype = tensor.dtype();
        let shape = tensor.shape();
        let data = tensor.data();
        let alignment_and_unit_size = dtype.size();

        let k = QuantizeKey {
            dtype: util::dtype_str(&dtype).to_owned(),
            shape: shape.to_vec(),
        };
        let Some(treatment) = plan.catalog.get(&k) else {
            anyhow::bail!("treatment of tensor type not described in plan: type= {k:?}");
        };

        let (dtype, alignment, data) = match treatment.treatment {
            QuantizeTreatment::keep => (
                util::dtype_str(&dtype),
                alignment_and_unit_size,
                data.into(),
            ),
            _ => {
                ensure!(dtype == Dtype::F32, "only F32 tensors can be quantized");
                let mut _loss = [0i64; 1 << 4];

                let n_elem = data.len() / alignment_and_unit_size;
                let mut work_buffer = vec![0u8; n_elem * 4];

                ensure!(data.len() % alignment_and_unit_size == 0);
                let ne_0 = *shape
                    .iter()
                    .find(|&&x| x != 1)
                    .with_context(|| "tensor has no dim that != 1")?;

                macro_rules! quantize {
                    ($func : ident) => {{
                        let cur_size = unsafe {
                            ggml_sys::$func(
                                std::mem::transmute(data.as_ptr()),
                                std::mem::transmute(work_buffer.as_mut_ptr()),
                                n_elem.try_into()?,
                                ne_0.try_into()?,
                                _loss.as_mut_ptr(),
                            )
                        };
                        work_buffer[0..cur_size].to_vec()
                    }};
                }

                match treatment.treatment {
                    QuantizeTreatment::keep => unreachable!(),

                    // == float32 delta/min  ==
                    //
                    QuantizeTreatment::q4_0 => {
                        let converted = quantize!(ggml_quantize_q4_0);
                        ("q4_0", 4, converted.into())
                    }
                    QuantizeTreatment::q4_1 => {
                        let converted = quantize!(ggml_quantize_q4_1);
                        ("q4_1", 4, converted.into())
                    }
                    QuantizeTreatment::q8_0 => {
                        let converted = quantize!(ggml_quantize_q8_0);
                        ("q8_0", 4, converted.into())
                    }

                    // == float16 delta/min ==
                    //
                    // QuantizeTreatment::q4_2 => {
                    //     let converted = quantize!(ggml_quantize_q4_2);
                    //     ("q4_2", 2, converted.into())
                    // }
                    QuantizeTreatment::q5_0 => {
                        let converted = quantize!(ggml_quantize_q5_0);
                        ("q5_0", 2, converted.into())
                    }
                    QuantizeTreatment::q5_1 => {
                        let converted = quantize!(ggml_quantize_q5_1);
                        ("q5_1", 2, converted.into())
                    }
                }
            }
        };
        let offsets = write_plan.plan_write(alignment, data);
        out.insert(
            name,
            json!({ "dtype": dtype, "shape": shape, "offsets": offsets }),
        );
    }

    let mut writer = File::create(args.model_out)?;
    let header = serde_json::to_vec(&out)?;
    let header_len = next_multiple_of(header.len(), 8);
    let padding = header_len - header.len();
    writer.write_all(&(header_len as u64).to_le_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(&vec![0u8; padding])?;

    for a in write_plan.tensors {
        writer.write_all(&vec![0u8; a.leading_padding])?;
        writer.write_all(&a.data)?;
    }

    Ok(())
}

#[derive(Default)]
struct TensorPlaner<'a> {
    /// first free byte
    free_offset: usize,
    tensors: Vec<TensorPlanerTensor<'a>>,
}

struct TensorPlanerTensor<'a> {
    leading_padding: usize,
    data: Cow<'a, [u8]>,
}

impl<'a> TensorPlaner<'a> {
    fn plan_write(&mut self, alignment: usize, data: Cow<'a, [u8]>) -> (usize, usize) {
        let start = next_multiple_of(self.free_offset, alignment);
        let padding = start - self.free_offset;
        let end = start + data.len();
        self.tensors.push(TensorPlanerTensor {
            leading_padding: padding,
            data,
        });
        self.free_offset = end;
        (start, end)
    }
}

fn next_multiple_of(lhs: usize, multiple: usize) -> usize {
    match lhs % multiple {
        0 => lhs,
        r => lhs + (multiple - r),
    }
}
