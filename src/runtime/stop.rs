//! Stop a model on an existing host without killing processes or invoking a shell.
use crate::config::StopConfig;
use std::time::Duration;

pub(super) fn run(config: StopConfig) -> Result<(), String> {
    if !matches!(config.host.as_str(), "127.0.0.1" | "localhost" | "::1") {
        return Err("stop requires a loopback host; no remote host is contacted".into());
    }
    let hostname = if config.host == "::1" {
        "[::1]"
    } else {
        &config.host
    };
    let url = format!("http://{hostname}:{}/api/stop", config.port);
    let client = reqwest::blocking::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_secs(3))
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client
        .post(url)
        .json(&serde_json::json!({"model":config.model}));
    if let Some(key) = config.api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().map_err(|e| {
        format!(
            "cannot contact lma host: {}; use `lma sp -p PORT` (NSS: `lma nss stop --port PORT`) if the host is not on port 8080",
            e.without_url()
        )
    })?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "host refused stop ({status}): {}",
            response.text().unwrap_or_default()
        ));
    }
    println!("Model unloaded.");
    Ok(())
}
