use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use anyhow::Context;
use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use tokio::sync::Mutex;

use crate::{ExecutionMode, setup_execution_providers};

const CACHE_DIR: &str = "cache";

// Don't be too small for dynamic quantization
pub const BATCH_SIZE: usize = 128;

pub struct ModelInfo {
    pub id: String,
    pub revision: String,
    pub onnx_file: String,
    pub tokenizer_file: String,
    pub config_file: String,
    pub special_tokens_map_file: String,
    pub tokenizer_config_file: String,
    pub pooling: Pooling,
    #[allow(dead_code)]
    pub quantization: QuantizationMode,
    #[allow(dead_code)]
    pub embedding_dims: usize,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum ModelKind {
    #[default]
    Int8,
    Fp16,
    Fp32,
}

pub fn get_embedding_model_info(kind: ModelKind) -> ModelInfo {
    ModelInfo {
        id: "gety-ai/granite-embedding-107m-multilingual-onnx".into(),
        revision: "4713bc4053d578f45e263e5857262b69f1f4296f".into(),
        onnx_file: match kind {
            ModelKind::Int8 => "onnx/model_quint8_avx2.onnx".into(),
            ModelKind::Fp16 => "onnx/model_optimized.onnx".into(),
            ModelKind::Fp32 => "onnx/model.onnx".into(),
        },
        tokenizer_file: "tokenizer.json".into(),
        config_file: "config.json".into(),
        special_tokens_map_file: "special_tokens_map.json".into(),
        tokenizer_config_file: "tokenizer_config.json".into(),
        pooling: Pooling::Cls,
        quantization: QuantizationMode::Dynamic,
        embedding_dims: 384,
    }
}

pub struct Model {
    pub info: ModelInfo,
    cache: PathBuf,
    model: OnceLock<Arc<Mutex<TextEmbedding>>>,
    downloaded: tokio::sync::OnceCell<()>,
    profiling_output: Option<PathBuf>,
}

impl Model {
    pub fn new(info: ModelInfo, cache: &Path, profiling_output: Option<&Path>) -> Self {
        Self {
            info,
            cache: std::path::absolute(&cache).expect("cannot absolute cache path"),
            model: OnceLock::new(),
            downloaded: tokio::sync::OnceCell::new(),
            profiling_output: profiling_output
                .map(|p| std::path::absolute(p).expect("cannot absolute profiling output path")),
        }
    }

    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache
    }

    fn model_dir(&self) -> PathBuf {
        self.cache.join(format!(
            "models--{}/snapshots/{}",
            self.info.id.replace('/', "--"),
            self.info.revision
        ))
    }

    pub async fn download_model(&self) -> anyhow::Result<()> {
        self.downloaded
            .get_or_try_init(|| async {
                self.download_model_inner().await?;
                anyhow::Ok(())
            })
            .await?;
        Ok(())
    }

    async fn download_model_inner(&self) -> anyhow::Result<()> {
        let cache = hf_hub::Cache::new(self.cache_dir().to_path_buf());
        let api = hf_hub::api::tokio::ApiBuilder::from_cache(cache)
            .with_endpoint("https://hf-mirror.com".to_string())
            .build()
            .unwrap();

        // Create a repo reference and download the model file
        let repo = api.repo(hf_hub::Repo::with_revision(
            self.info.id.clone(),
            hf_hub::RepoType::Model,
            self.info.revision.clone(),
        ));
        repo.get(&self.info.onnx_file)
            .await
            .expect("Failed to download onnx file");
        repo.get(&self.info.config_file)
            .await
            .expect("Failed to download config file");
        repo.get(&self.info.tokenizer_file)
            .await
            .expect("Failed to download tokenizer file");
        repo.get(&self.info.special_tokens_map_file)
            .await
            .expect("Failed to download special tokens map file");
        repo.get(&self.info.tokenizer_config_file)
            .await
            .expect("Failed to download tokenizer config file");

        anyhow::Ok(())
    }

    fn get_model(
        &self,
        execution_mode: ExecutionMode,
    ) -> anyhow::Result<&Arc<Mutex<TextEmbedding>>> {
        self.model.get_or_try_init(|| {
            let dir = self.model_dir();
            let model = (|| {
                {
                    anyhow::Ok(UserDefinedEmbeddingModel::new(
                        fs::read(dir.join(&self.info.onnx_file))?,
                        TokenizerFiles {
                            tokenizer_file: fs::read(dir.join(&self.info.tokenizer_file))?,
                            config_file: fs::read(dir.join(&self.info.config_file))?,
                            special_tokens_map_file: fs::read(
                                dir.join(&self.info.special_tokens_map_file),
                            )?,
                            tokenizer_config_file: fs::read(
                                dir.join(&self.info.tokenizer_config_file),
                            )?,
                        },
                    ))
                }
            })()
            .context("Failed to read model files")?;
            let execution_providers = setup_execution_providers(execution_mode);
            let mut opts = InitOptionsUserDefined::new()
                .with_parallel_execution(false)
                .with_execution_providers(execution_providers);
            if let Some(profiling_output) = self.profiling_output.as_ref() {
                eprintln!("enable profiling, output: {:?}", profiling_output);
                opts = opts.with_profiling_output(profiling_output);
            }
            Ok(Arc::new(Mutex::new(
                TextEmbedding::try_new_from_user_defined(
                    model
                        .with_pooling(self.info.pooling.clone())
                        // FastEmbed is just checking QuantizationMode to raise error if it's dynamic and the size of batch inference is not equal to the size of chunks
                        .with_quantization(QuantizationMode::Static),
                    opts,
                )?,
            )))
        })
    }

    pub async fn embed_chunks(
        &self,
        chunks: Vec<String>,
        execution_mode: ExecutionMode,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        self.download_model().await?;

        let model = self.get_model(execution_mode)?.clone();
        let mut model_guard = model.lock_owned().await;
        let embeddings =
            tokio::task::spawn_blocking(move || model_guard.embed(chunks, Some(BATCH_SIZE)))
                .await
                .unwrap()?;

        Ok(embeddings)
    }
}
