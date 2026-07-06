use std::any::Any;

use anyhow::{Context, Result};
use dotenvy;
use hf_hub::HFClient;
async fn model_get() -> Result<()> {
    dotenvy::dotenv().ok();
    let hf_token = std::env::var("HF_TOKEN").context("HF_TOKEN没设置")?;
    let client = HFClient::builder()
        .token(hf_token)
        .build()
        .context("setup client with hf token")?;
    check_status(client).await?;
    Ok(())
}
async fn check_status(client: HFClient) -> Result<()> {
    println!("\n===whoami===");
    match client.whoami().send().await {
        Ok(user) => {
            let orgs: Vec<String> = user
                .orgs
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .filter_map(|b| b.name.clone())
                .collect();
            println!(
                "username:{},type:{:?},orgs:{:?}",
                user.username,
                user.type_id(),
                orgs
            );
            Ok(())
        }
        Err(err) => Err(anyhow::anyhow!("认证失败，请检查HFTOKEN是否正确:{}"), err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_check_status() {
        model_get().await;
    }
}
