use std::{
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use hf_hub::HFClient;
use safetensors::SafeTensors;
#[derive(Clone)]
pub struct Model {
    model_name: String,
    model_owner: String,
}
#[derive(Debug)]
pub struct TensorsRecord {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub numel: usize,
    pub size_bytes: usize,
    pub module_path: Vec<String>,
    pub kind: TensorKind,
}
pub enum TensorKind {
    Weight,
    Bias,
    LayerNorm,
    Attention,
    Embedding,
    Other,
}

impl Debug for TensorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorKind::Attention => write!(f, "Attention"),
            TensorKind::Embedding => write!(f, "Embedding"),
            TensorKind::LayerNorm => write!(f, "LayerNorm"),
            TensorKind::Weight => write!(f, "Weight"),
            TensorKind::Bias => write!(f, "Bias"),
            TensorKind::Other => write!(f, "Other"),
        }
    }
}
pub async fn client_get() -> Result<HFClient> {
    dotenvy::dotenv().ok();
    let hf_token = std::env::var("HF_TOKEN").context("HF_TOKEN没设置")?;
    let client = HFClient::builder()
        .token(hf_token)
        .build()
        .context("setup client with hf token")?;
    Ok(client)
}
pub async fn check_status(client: &HFClient) -> Result<()> {
    println!("\n===whoami===");
    let user = client
        .whoami()
        .send()
        .await
        .context("认证失败，请检查HFTOKEN")?;
    let orgs: Vec<String> = user
        .orgs
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|org| org.name.clone())
        .collect();
    println!(
        "username:{},type:{:?},orgs:{:?}",
        user.username, user.user_type, orgs
    );
    Ok(())
}

pub async fn download_safetensor(client: HFClient, m1: Model) -> Result<()> {
    let model = client.model(&m1.model_owner, &m1.model_name);
    let download_dir = PathBuf::from("hf-downloads/").join(m1.model_name);
    let path = model
        .download_file()
        .filename("model.safetensors")
        .local_dir(download_dir)
        .send()
        .await
        .context("下载模型失败")?;
    println!("模型下载到{:?}", path);
    Ok(())
}

pub fn inspect_safetensors(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let data =
        fs::read(path).with_context(|| format!("读取safetensors文件失败:{}", path.display()))?;
    let tensors = SafeTensors::deserialize(&data)
        .with_context(|| format!("解析safentensors文件失败:{}", path.display()))?;
    println!("== tensor in {} ==", path.display());
    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("读取 tensor metadata 失败:{name}"))?;
        println!(
            "{:<90} dtype={:?},shape={:?}",
            name,
            tensor.dtype(),
            tensor.shape()
        );
    }
    Ok(())
}
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<Vec<TensorsRecord>> {
    let path = path.as_ref();
    let data = fs::read(path).with_context(|| format!("模型文件{}打开错误", path.display()))?;
    let tensors = SafeTensors::deserialize(&data)
        .with_context(|| format!("解析safentensors文件失败:{}", path.display()))?;
    let mut records: Vec<TensorsRecord> = Vec::new();
    for name in tensors.names() {
        let tensor = tensors
            .tensor(name)
            .with_context(|| format!("读取metadata标签失败{}", name))?;
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();
        let numel = shape.iter().product();
        let size_bytes = numel * dtype.bitsize() / 8;
        let module_path: Vec<String> = name.split('.').map(String::from).collect();
        let kind = if module_path.iter().any(|p| p == "LayerNorm") {
            TensorKind::LayerNorm
        } else if module_path.iter().any(|p| p.contains("attention")) {
            TensorKind::Attention
        } else if module_path.iter().any(|p| p.contains("embedding")) {
            TensorKind::Embedding
        } else {
            match module_path.last().map(String::as_str) {
                Some("weight") => TensorKind::Weight,
                Some("bias") => TensorKind::Bias,
                _ => TensorKind::Other,
            }
        };
        records.push(TensorsRecord {
            name: name.to_string(),
            dtype: format!("{:?}", dtype),
            shape,
            numel,
            size_bytes,
            module_path,
            kind,
        });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_check_status() -> Result<()> {
        let client = client_get().await?;
        check_status(&client).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_download_safetensors() -> Result<()> {
        let m = Model {
            model_name: String::from("all-MiniLM-L12-v2"),
            model_owner: String::from("sentence-transformers"),
        };
        let client = client_get().await?;
        download_safetensor(client, m).await?;
        Ok(())
    }
    #[test]
    fn test_inspect() -> Result<()> {
        let path = "hf-downloads/all-MiniLM-L12-v2/model.safetensors";
        inspect_safetensors(path)?;
        Ok(())
    }
    #[test]
    fn test_load_safetensors() -> Result<()> {
        let path = "hf-downloads/all-MiniLM-L12-v2/model.safetensors";
        let tensors = load_safetensors(path)?;
        println!("{:?}", tensors);
        Ok(())
    }
}
