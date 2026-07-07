use std::path::PathBuf;

use anyhow::{Context, Ok, Result};
use dotenvy;
use hf_hub::HFClient;
struct Model {
    modelname: String,
    modelowner: String,
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

async fn download_safetensor(client: HFClient, model: Model) -> Result<()> {
    let model = client.model(model.modelowner, model.modelname);
    let path = model
        .download_file()
        .filename("model.safetensors")
        .local_dir(PathBuf::from("./hf-downloads"))
        .send()
        .await
        .context("下载模型失败")?;
    println!("模型下载到{:?}", path);
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
            modelname: String::from("albert-base-v2"),
            modelowner: String::from("albert"),
        };
        let client = client_get().await?;
        download_safetensor(client, m).await?;
        Ok(())
    }
}
