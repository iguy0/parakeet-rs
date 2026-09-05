use crate::config::{CtcConfig, ModelConfig};
use crate::error::{Error, Result};
use crate::execution::ModelConfig as ExecutionConfig;
use crate::tensor_utils::argmax_f32;
use ndarray::{Array2, Axis};
use ort::session::{Session, SessionOutputs};
use ort::value::TensorRef;
use std::collections::HashMap;
use std::path::Path;

pub struct ParakeetModel {
    session: Session,
    config: ModelConfig,
    ctc_config: CtcConfig,
    has_attention_mask: bool,
}

impl ParakeetModel {
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        Self::from_pretrained_with_config(model_path, ExecutionConfig::default())
    }

    pub fn from_pretrained_with_config<P: AsRef<Path>>(
        model_path: P,
        exec_config: ExecutionConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();

        // Use default config (hardcoded constants for Parakeet-CTC-0.6b: please see: json files https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX/tree/main)
        let mut config = ModelConfig::default();

        let session = exec_config.build_session(model_path)?;
        let ctc_config = ctc_config_from_session(&session);
        config.pad_token_id = ctc_config.blank_id as usize;

        let has_attention_mask = session
            .inputs()
            .iter()
            .any(|input| input.name() == "attention_mask");

        Ok(Self {
            session,
            config,
            ctc_config,
            has_attention_mask,
        })
    }
    pub fn forward(&mut self, features: Array2<f32>) -> Result<Array2<f32>> {
        let outputs = self.run_features(features)?;
        let (time_steps_out, vocab_size, data) = extract_logits_view(&outputs)?;
        Array2::from_shape_vec((time_steps_out, vocab_size), data.to_vec())
            .map_err(|e| Error::Model(format!("Failed to create array: {e}")))
    }

    /// Greedy argmax per encoder frame without materializing the full logit tensor.
    pub fn forward_frame_ids(&mut self, features: Array2<f32>) -> Result<Vec<u32>> {
        let outputs = self.run_features(features)?;
        let (time_steps_out, vocab_size, data) = extract_logits_view(&outputs)?;
        let mut ids = Vec::with_capacity(time_steps_out);
        for t in 0..time_steps_out {
            let row = &data[t * vocab_size..(t + 1) * vocab_size];
            ids.push(argmax_f32(row).0 as u32);
        }
        Ok(ids)
    }

    fn run_features(&mut self, features: Array2<f32>) -> Result<SessionOutputs<'_>> {
        let time_steps = features.shape()[0];
        let input = features.insert_axis(Axis(0));
        let input = if input.is_standard_layout() {
            input
        } else {
            input.as_standard_layout().into_owned()
        };
        let input_value = TensorRef::<f32>::from_array_view(input.view())?;

        if self.has_attention_mask {
            let attention_mask = Array2::<i64>::ones((1, time_steps));
            let attention_mask_value = TensorRef::<i64>::from_array_view(attention_mask.view())?;
            Ok(self.session.run(ort::inputs!(
                "input_features" => input_value,
                "attention_mask" => attention_mask_value
            ))?)
        } else {
            Ok(self
                .session
                .run(ort::inputs!("input_features" => input_value))?)
        }
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn ctc_config(&self) -> CtcConfig {
        self.ctc_config
    }

    pub fn has_attention_mask(&self) -> bool {
        self.has_attention_mask
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    pub fn pad_token_id(&self) -> usize {
        self.config.pad_token_id
    }
}

fn ctc_config_from_session(session: &Session) -> CtcConfig {
    let Ok(meta) = session.metadata() else {
        return CtcConfig::default();
    };
    let Ok(key_names) = meta.custom_keys() else {
        return CtcConfig::default();
    };

    let mut keys = HashMap::new();
    for name in key_names {
        if let Some(value) = meta.custom(&name) {
            keys.insert(name, value);
        }
    }
    CtcConfig::from_metadata(&keys)
}

fn extract_logits_view<'a>(
    outputs: &'a SessionOutputs<'_>,
) -> Result<(usize, usize, &'a [f32])> {
    let logits_value = &outputs["logits"];
    let (shape, data) = logits_value
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Model(format!("Failed to extract logits: {e}")))?;
    let shape_dims = shape.as_ref();
    if shape_dims.len() != 3 {
        return Err(Error::Model(format!(
            "Expected 3D logits, got shape: {shape_dims:?}"
        )));
    }
    let batch_size = shape_dims[0] as usize;
    if batch_size != 1 {
        return Err(Error::Model(format!(
            "Expected batch size 1, got {batch_size}"
        )));
    }
    Ok((shape_dims[1] as usize, shape_dims[2] as usize, data))
}
