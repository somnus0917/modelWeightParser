use std::path::PathBuf;

use anyhow::{Context, Result};
use futures::StreamExt;
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use hf_hub::repository::RepoTreeEntry;

const MODEL_OWNER: &str = "prajjwal1";
const MODEL_NAME: &str = "bert-tiny";
const DOWNLOAD_FILE: &str = "config.json";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // ---- 1. Client ---------------------------------------------------------
    // `HFClient::new()` reads HF_TOKEN / HF_ENDPOINT / HF_HOME / HF_HUB_CACHE
    // from the environment and configures the underlying reqwest client +
    // on-disk cache. `HFClient` is internally an Arc, so clones are cheap.
    let client = hf_hub::HFClient::new().context("building HF client from env")?;
    println!("== client ==");
    println!("  endpoint : {}", client.endpoint());
    println!("  cache    : {}", client.cache_dir().display());
    println!("  cached?  : {}", client.cache_enabled());

    // ---- 2. whoami ---------------------------------------------------------
    // Hits GET /api/whoami-v2. Returns Ok(User) for an anonymous caller too
    // (with the username "hf-less-anonymous" or similar), so the call
    // rarely fails on its own.
    println!("\n== whoami ==");
    match client.whoami().send().await {
        Ok(user) => {
            let orgs: Vec<String> = user
                .orgs
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .filter_map(|o| o.name.clone())
                .collect();
            println!(
                "  username = {}  type = {:?}  orgs = {:?}",
                user.username, user.user_type, orgs
            );
        }
        Err(e) => println!("  (skipped: {e})"),
    }

    // ---- 3. list_models ----------------------------------------------------
    // `list_models` returns an `impl Stream<Item = HFResult<ModelInfo>>` —
    // the client paginates the /api/models endpoint transparently. We
    // collect a few entries sorted by downloads to show the API shape.
    println!("\n== top text-generation models ==");
    {
        let stream = client
            .list_models()
            .sort("downloads")
            .pipeline_tag("text-generation")
            .limit(5)
            .send()
            .context("opening list_models stream")?;
        futures::pin_mut!(stream);
        let mut shown = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(m) => {
                    shown += 1;
                    println!(
                        "  {shown:>2}. {:<40}  downloads={:<8} likes={}",
                        m.id,
                        m.downloads.unwrap_or(0),
                        m.likes.unwrap_or(0),
                    );
                }
                Err(e) => println!("  (item error: {e})"),
            }
        }
    }

    // ---- 4. model info -----------------------------------------------------
    // The typed handle binds the repo kind once. `info()` issues
    // GET /api/models/{owner}/{name} and returns a fully-typed `ModelInfo`.
    let model = client.model(MODEL_OWNER, MODEL_NAME);
    let info = model
        .info()
        .send()
        .await
        .with_context(|| format!("fetching info for {MODEL_OWNER}/{MODEL_NAME}"))?;
    println!("\n== model info: {MODEL_OWNER}/{MODEL_NAME} ==");
    println!("  id           : {}", info.id);
    println!("  author       : {:?}", info.author);
    println!("  pipeline_tag : {:?}", info.pipeline_tag);
    println!("  library      : {:?}", info.library_name);
    println!("  downloads    : {:?}", info.downloads);
    println!("  likes        : {:?}", info.likes);
    println!(
        "  tags (first) : {:?}",
        info.tags.as_ref().and_then(|t| t.first())
    );
    println!("  sha          : {:?}", info.sha);
    println!("  last modified: {:?}", info.last_modified);

    // ---- 5. list_tree ------------------------------------------------------
    // Enumerate every file in the repo. `recursive(true)` walks subfolders;
    // the result is a stream so very large repos don't have to fit in memory.
    // Each entry is a `RepoTreeEntry::{File, Directory}` enum variant.
    println!("\n== tree of {MODEL_OWNER}/{MODEL_NAME} ==");
    {
        let stream = model
            .list_tree()
            .recursive(true)
            .send()
            .context("opening list_tree stream")?;
        futures::pin_mut!(stream);
        let mut count = 0;
        while let Some(item) = stream.next().await {
            match item {
                Ok(RepoTreeEntry::File { path, size, .. }) => {
                    count += 1;
                    println!("  {count:>3}. {path:<32}  size={size}");
                }
                Ok(RepoTreeEntry::Directory { path, .. }) => {
                    count += 1;
                    println!("  {count:>3}. {path:<32}  (dir)");
                }
                Err(e) => println!("  (item error: {e})"),
            }
        }
        println!("  ({count} entries)");
    }

    // ---- 6. download_file --------------------------------------------------
    // Downloads go through the content-addressed cache by default. A second
    // call with the same filename will return the cached path instantly.
    // The optional `Progress` callback is invoked from another task — we use
    // a tiny in-memory ring buffer so the demo doesn't need a TUI dependency.
    println!("\n== download {DOWNLOAD_FILE} ==");
    let progress = make_progress_logger();
    let path: PathBuf = model
        .download_file()
        .filename(DOWNLOAD_FILE)
        .progress(progress)
        .send()
        .await
        .context("downloading config.json")?;
    println!("  cached at : {}", path.display());

    // Read it back and show the first 400 bytes — proves the file landed
    // on disk and is readable by normal Rust APIs (no HF lock-in required).
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let snippet: String = bytes.iter().take(400).map(|b| char::from(*b)).collect();
    println!("  size      : {} bytes", bytes.len());
    println!(
        "  preview   : {snippet}{}",
        if bytes.len() > 400 { "…" } else { "" }
    );

    // ---- 7. scan_cache -----------------------------------------------------
    // `scan_cache` walks HF_HOME / HF_HUB_CACHE and returns everything the
    // library has stored locally, grouped by repo and revision. Handy for
    // a "show me what's already downloaded" view.
    println!("\n== local cache ==");
    let info = client
        .scan_cache()
        .send()
        .await
        .context("scanning HF cache")?;
    if info.repos.is_empty() {
        println!("  (empty)");
    } else {
        println!("  {} cached repo(s):", info.repos.len());
        for repo in info.repos.iter().take(8) {
            println!(
                "    {:<50}  size={} bytes  revisions={}",
                repo.repo_id,
                repo.size_on_disk,
                repo.revisions.len()
            );
        }
        if info.repos.len() > 8 {
            println!("    …and {} more", info.repos.len() - 8);
        }
    }
    println!("\ndone.");
    Ok(())
}

fn make_progress_logger() -> Progress {
    struct Logger;
    impl ProgressHandler for Logger {
        fn on_progress(&self, event: &ProgressEvent) {
            if let ProgressEvent::Download(DownloadEvent::Start {
                total_files,
                total_bytes,
            }) = event
            {
                println!("  > start   {total_files} file(s), {total_bytes} bytes");
            }
        }
    }
    Progress::new(Logger)
}
