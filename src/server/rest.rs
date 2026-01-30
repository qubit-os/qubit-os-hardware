// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! REST API server implementation using axum.
//!
//! Provides a REST facade over the gRPC service for easier integration
//! with web clients and debugging.
//!
//! # Endpoints
//!
//! - `GET /api/v1/health` - Health check
//! - `GET /api/v1/backends` - List backends
//! - `GET /api/v1/backends/{name}` - Get backend info
//! - `POST /api/v1/execute` - Execute a pulse
//! - `GET /api/v1/version` - Get server version

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{debug, error, info};
use uuid::Uuid;

use super::ServerState;
use crate::backend::ExecutePulseRequest as BackendRequest;
use crate::config::ServerConfig;
use crate::error::{Error, Result};

/// REST server for the HAL.
pub struct RestServer {
    state: Arc<ServerState>,
}

impl RestServer {
    /// Create a new REST server.
    pub fn new(state: Arc<ServerState>) -> Self {
        Self { state }
    }

    /// Start the REST server.
    pub async fn serve(self, config: &ServerConfig) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", config.host, config.rest_port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid REST address: {}", e)))?;

        info!(address = %addr, "Starting REST server");

        // Build router
        let app = Router::new()
            .route("/api/v1/health", get(health_check))
            .route("/api/v1/backends", get(list_backends))
            .route("/api/v1/backends/:name", get(get_backend_info))
            .route("/api/v1/execute", post(execute_pulse))
            .route("/api/v1/version", get(get_version))
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
            .layer(TraceLayer::new_for_http())
            .with_state(self.state.clone());

        let mut shutdown_rx = self.state.shutdown_receiver();

        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .map_err(|e| Error::Server(format!("Failed to bind REST server: {}", e)))?;

        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.changed().await;
                info!("REST server shutting down");
            })
            .await
            .map_err(|e| Error::Server(format!("REST server error: {}", e)))?;

        Ok(())
    }
}

// =============================================================================
// Request/Response types
// =============================================================================

/// Health check response.
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    backends: Vec<BackendHealth>,
}

/// Individual backend health.
#[derive(Debug, Serialize)]
struct BackendHealth {
    name: String,
    status: String,
    backend_type: String,
}

/// Backend list response.
#[derive(Debug, Serialize)]
struct BackendsResponse {
    backends: Vec<BackendSummary>,
    default_backend: Option<String>,
}

/// Backend summary.
#[derive(Debug, Serialize)]
struct BackendSummary {
    name: String,
    backend_type: String,
    num_qubits: u32,
}

/// Backend info response.
#[derive(Debug, Serialize)]
struct BackendInfoResponse {
    name: String,
    backend_type: String,
    tier: String,
    num_qubits: u32,
    available_qubits: Vec<u32>,
    supported_gates: Vec<String>,
    supports_state_vector: bool,
    supports_noise_model: bool,
    software_version: String,
}

/// Execute pulse request.
#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    pulse_id: Option<String>,
    backend_name: Option<String>,
    i_envelope: Vec<f64>,
    q_envelope: Vec<f64>,
    duration_ns: u32,
    target_qubits: Vec<u32>,
    num_shots: u32,
    measurement_basis: Option<String>,
    return_state_vector: Option<bool>,
    include_noise: Option<bool>,
}

/// Execute pulse response.
#[derive(Debug, Serialize)]
struct ExecuteResponse {
    request_id: String,
    pulse_id: String,
    bitstring_counts: std::collections::HashMap<String, u32>,
    total_shots: u32,
    successful_shots: u32,
    fidelity_estimate: Option<f64>,
    state_vector: Option<Vec<(f64, f64)>>,
}

/// Version response.
#[derive(Debug, Serialize)]
struct VersionResponse {
    version: String,
    name: String,
}

/// Error response.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
}

// =============================================================================
// Handlers
// =============================================================================

/// Health check endpoint.
async fn health_check(
    State(state): State<Arc<ServerState>>,
) -> std::result::Result<Json<HealthResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut backends = Vec::new();
    let mut overall_healthy = true;

    for name in state.registry.list() {
        if let Ok(backend) = state.registry.get(&name) {
            let status = match backend.health_check().await {
                Ok(s) => {
                    let status_str = match s {
                        crate::backend::HealthStatus::Healthy => "healthy",
                        crate::backend::HealthStatus::Degraded => {
                            overall_healthy = false;
                            "degraded"
                        }
                        crate::backend::HealthStatus::Unavailable => {
                            overall_healthy = false;
                            "unavailable"
                        }
                    };
                    status_str.to_string()
                }
                Err(e) => {
                    overall_healthy = false;
                    format!("error: {}", e)
                }
            };

            backends.push(BackendHealth {
                name: name.clone(),
                status,
                backend_type: backend.backend_type().to_string(),
            });
        }
    }

    Ok(Json(HealthResponse {
        status: if overall_healthy {
            "healthy"
        } else {
            "degraded"
        }
        .to_string(),
        backends,
    }))
}

/// List backends endpoint.
async fn list_backends(
    State(state): State<Arc<ServerState>>,
) -> std::result::Result<Json<BackendsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut backends = Vec::new();

    for name in state.registry.list() {
        if let Ok(backend) = state.registry.get(&name) {
            if let Ok(info) = backend.get_hardware_info().await {
                backends.push(BackendSummary {
                    name: info.name,
                    backend_type: info.backend_type.to_string(),
                    num_qubits: info.num_qubits,
                });
            }
        }
    }

    Ok(Json(BackendsResponse {
        backends,
        default_backend: state.registry.default_backend_name(),
    }))
}

/// Get backend info endpoint.
async fn get_backend_info(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> std::result::Result<Json<BackendInfoResponse>, (StatusCode, Json<ErrorResponse>)> {
    let backend = state.registry.get(&name).map_err(|e| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "BACKEND_NOT_FOUND".to_string(),
            }),
        )
    })?;

    let info = backend.get_hardware_info().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "HARDWARE_INFO_ERROR".to_string(),
            }),
        )
    })?;

    Ok(Json(BackendInfoResponse {
        name: info.name,
        backend_type: info.backend_type.to_string(),
        tier: info.tier,
        num_qubits: info.num_qubits,
        available_qubits: info.available_qubits,
        supported_gates: info.supported_gates,
        supports_state_vector: info.supports_state_vector,
        supports_noise_model: info.supports_noise_model,
        software_version: info.software_version,
    }))
}

/// Execute pulse endpoint.
async fn execute_pulse(
    State(state): State<Arc<ServerState>>,
    Json(req): Json<ExecuteRequest>,
) -> std::result::Result<Json<ExecuteResponse>, (StatusCode, Json<ErrorResponse>)> {
    let request_id = Uuid::new_v4().to_string();
    let pulse_id = req.pulse_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    debug!(
        request_id = %request_id,
        pulse_id = %pulse_id,
        backend = ?req.backend_name,
        "REST execute_pulse request"
    );

    // Get backend
    let backend = state
        .registry
        .get_or_default(req.backend_name.as_deref())
        .map_err(|e| {
            (
                StatusCode::NOT_FOUND,
                Json(ErrorResponse {
                    error: e.to_string(),
                    code: "BACKEND_NOT_FOUND".to_string(),
                }),
            )
        })?;

    // Build request
    let num_time_steps = req.i_envelope.len() as u32;
    let backend_request = BackendRequest {
        pulse_id: pulse_id.clone(),
        i_envelope: req.i_envelope,
        q_envelope: req.q_envelope,
        duration_ns: req.duration_ns,
        num_time_steps,
        target_qubits: req.target_qubits,
        num_shots: req.num_shots,
        measurement_basis: req.measurement_basis.unwrap_or_else(|| "Z".to_string()),
        return_state_vector: req.return_state_vector.unwrap_or(false),
        include_noise: req.include_noise.unwrap_or(false),
    };

    // Execute
    let result = backend.execute_pulse(backend_request).await.map_err(|e| {
        error!(error = %e, "Pulse execution failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
                code: "EXECUTION_ERROR".to_string(),
            }),
        )
    })?;

    Ok(Json(ExecuteResponse {
        request_id,
        pulse_id,
        bitstring_counts: result.bitstring_counts,
        total_shots: result.total_shots,
        successful_shots: result.successful_shots,
        fidelity_estimate: result.fidelity_estimate,
        state_vector: result.state_vector,
    }))
}

/// Get version endpoint.
async fn get_version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        name: "QubitOS HAL".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRegistry;

    #[tokio::test]
    async fn test_rest_server_creation() {
        let registry = Arc::new(BackendRegistry::default());
        let state = Arc::new(ServerState::new(registry));
        let _server = RestServer::new(state);
        // Server created successfully - if we got here without panic, test passed
    }
}
