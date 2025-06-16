use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use std::sync::Arc;

use crate::config_parser::{AuthConfig, Config};
use crate::error::{Error, ErrorDetails};
use crate::gateway_util::AppState;

pub async fn auth_middleware(
    State(app_state): State<AppState>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let config = &app_state.config;

    // Skip authentication if disabled
    if !config.gateway.auth.enabled {
        return Ok(next.run(request).await);
    }

    // Skip authentication for health and status endpoints
    let path = request.uri().path();
    if path == "/health" || path == "/status" {
        return Ok(next.run(request).await);
    }

    // Check for Authorization header
    let auth_header = request
        .headers()
        .get("authorization")
        .or_else(|| request.headers().get("Authorization"));

    let auth_header = match auth_header {
        Some(header) => header,
        None => {
            tracing::warn!("Missing Authorization header");
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // Parse Bearer token
    let auth_str = match auth_header.to_str() {
        Ok(s) => s,
        Err(_) => {
            tracing::warn!("Invalid Authorization header format");
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    if !auth_str.starts_with("Bearer ") {
        tracing::warn!("Authorization header must use Bearer scheme");
        return Err(StatusCode::UNAUTHORIZED);
    }

    let token = &auth_str[7..]; // Remove "Bearer " prefix

    // Validate against static API keys
    if !config.gateway.auth.static_api_keys.contains(&token.to_string()) {
        tracing::warn!("Invalid API key provided");
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Token is valid, proceed with request
    Ok(next.run(request).await)
}