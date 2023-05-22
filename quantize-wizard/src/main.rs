#![feature(hash_raw_entry)]

mod util;

use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::{borrow::Cow, io::Write};

use anyhow::{ensure, Context};
use clap::Parser;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Safetensors Interactive Quantization Tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
enum Args {
    Plan(PlanArgs),
    Convert(ConvertArgs),
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
struct ConvertArgs {
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
        Args::Plan(args) => do_plan(args),
        Args::Convert(args) => do_convert(args),
    }
}

#[derive(Serialize, Deserialize)]
struct Plan {
    metadata: Option<HashMap<String, String>>,
    catalog: HashMap<QuantizeKey, QuantizeInfo>,
}

#[derive(Hash, PartialEq, Eq, Serialize, Deserialize, Debug)]
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
        let (_, info) = tensors
            .raw_entry_mut()
            .from_key(&k)
            .or_insert(k, Default::default());
        info.count += 1;
    }

    let writer = File::create(args.plan_out)?;
    cyrly::to_writer(
        &writer,
        &Plan {
            metadata,
            catalog: tensors,
        },
    )?;

    Ok(())
}

fn do_convert(args: ConvertArgs) -> anyhow::Result<()> {
    let plan_file = File::open(args.plan)?;
    let plan: Plan = serde_yaml::from_reader(plan_file)?;
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
