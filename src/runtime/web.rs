//! Public, dependency-free chat shell. API authentication remains in `server`.
//!
//! Mount `index` at `/`, `script` at `/chat.js`, and `style` at `/style.css`
//! outside the API authorization middleware.

use axum::{
    http::{HeaderValue, header},
    response::{Html, IntoResponse, Response},
};

const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'";
const INDEX: &str = include_str!("ui/index.html");
const SCRIPT: &str = include_str!("ui/chat.js");
const STYLE: &str = include_str!("ui/style.css");

fn secure(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CSP),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    // The assets are embedded in the executable, with no fingerprinted URLs.
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

pub(super) async fn index() -> impl IntoResponse {
    secure(Html(INDEX).into_response())
}

pub(super) async fn script() -> impl IntoResponse {
    secure(
        (
            [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
            SCRIPT,
        )
            .into_response(),
    )
}

pub(super) async fn style() -> impl IntoResponse {
    secure(([(header::CONTENT_TYPE, "text/css; charset=utf-8")], STYLE).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
        routing::get,
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn app() -> Router {
        Router::new()
            .route("/", get(index))
            .route("/chat.js", get(script))
            .route("/style.css", get(style))
    }

    #[tokio::test]
    async fn public_assets_have_exact_bodies_types_and_security_headers() {
        for (path, content_type, body) in [
            ("/", "text/html; charset=utf-8", INDEX),
            ("/chat.js", "text/javascript; charset=utf-8", SCRIPT),
            ("/style.css", "text/css; charset=utf-8", STYLE),
        ] {
            let response = app()
                .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let headers = response.headers();
            assert_eq!(headers[header::CONTENT_TYPE], content_type);
            assert_eq!(headers[header::CONTENT_SECURITY_POLICY], CSP);
            assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
            assert_eq!(headers[header::REFERRER_POLICY], "no-referrer");
            assert_eq!(headers[header::CACHE_CONTROL], "no-store");
            assert!(!headers.contains_key(header::SET_COOKIE));
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert_eq!(bytes.as_ref(), body.as_bytes());
        }
    }

    #[tokio::test]
    async fn head_requests_have_headers_without_asset_bodies() {
        for path in ["/", "/chat.js", "/style.css"] {
            let response = app()
                .oneshot(
                    Request::builder()
                        .method("HEAD")
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(response.headers()[header::CONTENT_SECURITY_POLICY], CSP);
            assert!(
                response
                    .into_body()
                    .collect()
                    .await
                    .unwrap()
                    .to_bytes()
                    .is_empty()
            );
        }
    }

    #[test]
    fn shell_uses_local_external_assets_and_strict_csp() {
        assert!(INDEX.contains("<script type=\"module\" src=\"/chat.js\"></script>"));
        assert!(INDEX.contains("href=\"/style.css\""));
        assert!(!INDEX.contains("style=\""));
        assert!(!INDEX.contains("onclick="));
        assert!(!CSP.contains("unsafe-inline"));
        assert!(!CSP.contains("unsafe-eval"));
        for directive in [
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self'",
            "img-src 'self' data: blob:",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "frame-ancestors 'none'",
        ] {
            assert!(CSP.split("; ").any(|value| value == directive));
        }
    }
}
