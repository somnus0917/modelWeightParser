use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Ok, Result};
use dotenvy;
use hf_hub::HFClient;
use safetensors::SafeTensors;
#[derive(Clone)]
struct Model {
    model_name: String,
    model_owner: String,
}
async fn client_get() -> Result<HFClient> {
    dotenvy::dotenv().ok();
    let hf_token = std::env::var("HF_TOKEN").context("HF_TOKEN没设置")?;
    let client = HFClient::builder()
        .token(hf_token)
        .build()
        .context("setup client with hf token")?;
    Ok(client)
}
async fn check_status(client: &HFClient) -> Result<()> {
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

async fn download_safetensor(client: HFClient, m1: Model) -> Result<()> {
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

fn inspect_safetensors(path: impl AsRef<Path>) -> Result<()> {
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
            model_name: String::from("albert-base-v2"),
            model_owner: String::from("albert"),
        };
        let client = client_get().await?;
        download_safetensor(client, m).await?;
        Ok(())
    }
    // #[test]
    // fn test_inspect() -> Result<()> {
    //     let path = "hf-downloads/model.safetensors";
    //     inspect_safetensors(path);
    //     Ok(())
    // }
}
