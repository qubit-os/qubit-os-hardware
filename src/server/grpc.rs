// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! gRPC server implementation using tonic.
//!
//! Implements the QuantumBackend gRPC service defined in qubit-os-proto.
//!
//! # Security
//!
//! - All inputs are validated before processing (envelope size, qubit bounds, etc.)
//! - Error messages are sanitized to not leak internal details

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

use super::ServerState;
use crate::backend::ExecutePulseRequest as BackendRequest;
use crate::config::ServerConfig;
use crate::error::{Error, Result};
use crate::proto::quantum::backend::v1::{
    quantum_backend_server::{QuantumBackend, QuantumBackendServer},
    BitstringCounts, ExecutePulseRequest, ExecutePulseResponse, GetHardwareInfoRequest,
    GetHardwareInfoResponse, HardwareInfo, HealthCheckRequest, HealthCheckResponse,
    HealthStatus as ProtoHealthStatus, MeasurementResult as ProtoMeasurementResult,
};
use crate::validation::{validate_api_request, MAX_QUBITS};

/// gRPC server for the QuantumBackend service.
pub struct GrpcServer {
    /// Shared server state
    state: Arc<ServerState>,
}

impl GrpcServer {
    /// Create a new gRPC server.
    pub fn new(state: Arc<ServerState>) -> Self {
        Self { state }
    }

    /// Start the gRPC server.
    pub async fn serve(self, config: &ServerConfig) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", config.host, config.grpc_port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid gRPC address: {}", e)))?;

        info!(address = %addr, "Starting gRPC server");

        let service = QuantumBackendService {
            state: self.state.clone(),
        };

        let mut shutdown_rx = self.state.shutdown_receiver();

        tonic::transport::Server::builder()
            .add_service(QuantumBackendServer::new(service))
            .serve_with_shutdown(addr, async {
                let _ = shutdown_rx.changed().await;
                info!("gRPC server shutting down");
            })
            .await
            .map_err(|e| Error::Server(format!("gRPC server error: {}", e)))?;

        Ok(())
    }
}

/// gRPC service implementation.
#[derive(Clone)]
struct QuantumBackendService {
    state: Arc<ServerState>,
}

/// Sanitize error message for gRPC response.
/// In production, we don't want to leak internal details.
fn sanitize_error_for_status(error: &Error) -> String {
    // For validation errors, the message is safe to show
    if let Error::Validation(ref ve) = error {
        return ve.to_string();
    }

    // For other errors, return a generic message in production
    #[cfg(debug_assertions)]
    {
        error.to_string()
    }

    #[cfg(not(debug_assertions))]
    {
        match error {
            Error::Backend(_) => "Backend execution error".to_string(),
            Error::Config(_) => "Configuration error".to_string(),
            Error::Server(_) => "Internal server error".to_string(),
            Error::Validation(ve) => ve.to_string(),
            Error::Io(_) => "I/O error".to_string(),
            Error::Serialization(_) => "Serialization error".to_string(),
        }
    }
}

#[tonic::async_trait]
impl QuantumBackend for QuantumBackendService {
    #[instrument(skip(self, request), fields(pulse_id, backend))]
    async fn execute_pulse(
        &self,
        request: Request<ExecutePulseRequest>,
    ) -> std::result::Result<Response<ExecutePulseResponse>, Status> {
        let req = request.into_inner();
        let request_id = Uuid::new_v4().to_string();

        debug!(
            request_id = %request_id,
            pulse_id = %req.pulse_id,
            backend = ?req.backend_name,
            envelope_len = req.i_envelope.len(),
            num_shots = req.num_shots,
            "Received ExecutePulse request"
        );

        // =====================================================================
        // SECURITY: Validate all inputs BEFORE any processing
        // =====================================================================

        // Get backend first to know max_qubits
        let backend = self
            .state
            .registry
            .get_or_default(req.backend_name.as_deref())
            .map_err(|e| {
                error!(error = %e, "Backend not found");
                Status::not_found(e.to_string())
            })?;

        // Get backend limits
        let backend_info = backend.get_hardware_info().await.map_err(|e| {
            error!(error = %e, "Failed to get backend info for validation");
            Status::internal("Failed to get backend info")
        })?;

        // Validate all inputs
        let max_qubits = std::cmp::min(backend_info.num_qubits, MAX_QUBITS);
        if let Err(e) = validate_api_request(
            &req.i_envelope,
            &req.q_envelope,
            req.num_shots,
            req.duration_ns,
            &req.target_qubits,
            max_qubits,
        ) {
            warn!(
                request_id = %request_id,
                error = %e,
                "Request validation failed"
            );
            return Err(Status::invalid_argument(sanitize_error_for_status(&e)));
        }

        // =====================================================================
        // Build and execute request
        // =====================================================================

        let backend_request = BackendRequest {
            pulse_id: req.pulse_id.clone(),
            i_envelope: req.i_envelope,
            q_envelope: req.q_envelope,
            duration_ns: req.duration_ns,
            num_time_steps: req.num_time_steps,
            target_qubits: req.target_qubits,
            num_shots: req.num_shots,
            measurement_basis: req.measurement_basis,
            return_state_vector: req.return_state_vector,
            include_noise: req.include_noise,
        };

        // Execute
        let result = backend.execute_pulse(backend_request).await.map_err(|e| {
            error!(error = %e, "Pulse execution failed");
            Status::from(Error::from(e))
        })?;

        // Convert result
        let response = ExecutePulseResponse {
            request_id,
            pulse_id: req.pulse_id,
            result: Some(ProtoMeasurementResult {
                bitstring_counts: Some(BitstringCounts {
                    counts: result
                        .bitstring_counts
                        .into_iter()
                        .map(|(k, v)| (k, v as i64))
                        .collect(),
                }),
                total_shots: result.total_shots,
                successful_shots: result.successful_shots,
                fidelity_estimate: result.fidelity_estimate,
                state_vector_real: result
                    .state_vector
                    .as_ref()
                    .map(|sv| sv.iter().map(|(r, _)| *r).collect())
                    .unwrap_or_default(),
                state_vector_imag: result
                    .state_vector
                    .as_ref()
                    .map(|sv| sv.iter().map(|(_, i)| *i).collect())
                    .unwrap_or_default(),
            }),
            error: None,
        };

        Ok(Response::new(response))
    }

    #[instrument(skip(self, request), fields(backend))]
    async fn get_hardware_info(
        &self,
        request: Request<GetHardwareInfoRequest>,
    ) -> std::result::Result<Response<GetHardwareInfoResponse>, Status> {
        let req = request.into_inner();

        debug!(backend = ?req.backend_name, "Received GetHardwareInfo request");

        // Get backend
        let backend = self
            .state
            .registry
            .get_or_default(req.backend_name.as_deref())
            .map_err(|e| Status::not_found(e.to_string()))?;

        // Get info
        let info = backend.get_hardware_info().await.map_err(|e| {
            error!(error = %e, "Failed to get hardware info");
            Status::from(Error::from(e))
        })?;

        let response = GetHardwareInfoResponse {
            info: Some(HardwareInfo {
                name: info.name,
                backend_type: match info.backend_type {
                    crate::backend::BackendType::Simulator => 0,
                    crate::backend::BackendType::Hardware => 1,
                },
                tier: info.tier,
                num_qubits: info.num_qubits,
                available_qubits: info.available_qubits,
                supported_gates: info.supported_gates,
                supports_state_vector: info.supports_state_vector,
                supports_noise_model: info.supports_noise_model,
                software_version: info.software_version,
            }),
        };

        Ok(Response::new(response))
    }

    #[instrument(skip(self, request), fields(backend))]
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        let req = request.into_inner();

        debug!(backend = ?req.backend_name, "Received HealthCheck request");

        // If specific backend requested, check it
        if let Some(ref name) = req.backend_name {
            let backend = self
                .state
                .registry
                .get(name)
                .map_err(|e| Status::not_found(e.to_string()))?;

            let status = backend.health_check().await.map_err(|e| {
                error!(error = %e, "Health check failed");
                Status::from(Error::from(e))
            })?;

            let proto_status = match status {
                crate::backend::HealthStatus::Healthy => ProtoHealthStatus::Healthy,
                crate::backend::HealthStatus::Degraded => ProtoHealthStatus::Degraded,
                crate::backend::HealthStatus::Unavailable => ProtoHealthStatus::Unavailable,
            };

            return Ok(Response::new(HealthCheckResponse {
                status: proto_status as i32,
                message: String::new(),
                backends: HashMap::new(),
            }));
        }

        // Check all backends
        let mut backends = HashMap::new();
        let mut overall_status = ProtoHealthStatus::Healthy;

        for name in self.state.registry.list() {
            if let Ok(backend) = self.state.registry.get(&name) {
                match backend.health_check().await {
                    Ok(status) => {
                        let proto_status = match status {
                            crate::backend::HealthStatus::Healthy => ProtoHealthStatus::Healthy,
                            crate::backend::HealthStatus::Degraded => {
                                if overall_status == ProtoHealthStatus::Healthy {
                                    overall_status = ProtoHealthStatus::Degraded;
                                }
                                ProtoHealthStatus::Degraded
                            }
                            crate::backend::HealthStatus::Unavailable => {
                                overall_status = ProtoHealthStatus::Degraded;
                                ProtoHealthStatus::Unavailable
                            }
                        };
                        backends.insert(name, proto_status as i32);
                    }
                    Err(e) => {
                        warn!(backend = %name, error = %e, "Backend health check failed");
                        backends.insert(name, ProtoHealthStatus::Unavailable as i32);
                        overall_status = ProtoHealthStatus::Degraded;
                    }
                }
            }
        }

        Ok(Response::new(HealthCheckResponse {
            status: overall_status as i32,
            message: String::new(),
            backends,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRegistry;
    use crate::proto::quantum::backend::v1::quantum_backend_server::QuantumBackend as QBTrait;
    use crate::test_utils::{DegradedMockBackend, MockBackend};

    fn empty_state() -> Arc<ServerState> {
        Arc::new(ServerState::new(Arc::new(BackendRegistry::default())))
    }

    fn state_with_mock() -> Arc<ServerState> {
        let registry = Arc::new(BackendRegistry::default());
        registry.register(MockBackend::simulator("test"));
        Arc::new(ServerState::new(registry))
    }

    fn make_service(state: Arc<ServerState>) -> QuantumBackendService {
        QuantumBackendService { state }
    }

    fn valid_pulse_request() -> ExecutePulseRequest {
        ExecutePulseRequest {
            pulse_id: "p1".to_string(),
            backend_name: None,
            i_envelope: vec![0.1; 10],
            q_envelope: vec![0.1; 10],
            duration_ns: 100,
            num_time_steps: 10,
            target_qubits: vec![0],
            num_shots: 1000,
            measurement_basis: "Z".to_string(),
            return_state_vector: false,
            include_noise: false,
        }
    }

    #[tokio::test]
    async fn test_grpc_server_creation() {
        let registry = Arc::new(BackendRegistry::default());
        let state = Arc::new(ServerState::new(registry));
        let _server = GrpcServer::new(state);
    }

    #[tokio::test]
    async fn test_execute_pulse_success() {
        let svc = make_service(state_with_mock());
        let resp = svc
            .execute_pulse(Request::new(valid_pulse_request()))
            .await
            .unwrap();

        let inner = resp.into_inner();
        assert!(!inner.request_id.is_empty());
        assert_eq!(inner.pulse_id, "p1");
        let result = inner.result.unwrap();
        assert_eq!(result.total_shots, 1000);
        assert_eq!(result.successful_shots, 1000);
    }

    #[tokio::test]
    async fn test_execute_pulse_no_backend() {
        let svc = make_service(empty_state());
        let err = svc
            .execute_pulse(Request::new(valid_pulse_request()))
            .await
            .unwrap_err();

        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_execute_pulse_validation_error() {
        let svc = make_service(state_with_mock());
        let mut req = valid_pulse_request();
        req.num_shots = 0; // invalid

        let err = svc.execute_pulse(Request::new(req)).await.unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_get_hardware_info_success() {
        let svc = make_service(state_with_mock());
        let resp = svc
            .get_hardware_info(Request::new(GetHardwareInfoRequest {
                backend_name: Some("test".to_string()),
            }))
            .await
            .unwrap();

        let info = resp.into_inner().info.unwrap();
        assert_eq!(info.name, "test");
        assert_eq!(info.num_qubits, 2);
    }

    #[tokio::test]
    async fn test_get_hardware_info_not_found() {
        let svc = make_service(state_with_mock());
        let err = svc
            .get_hardware_info(Request::new(GetHardwareInfoRequest {
                backend_name: Some("nope".to_string()),
            }))
            .await
            .unwrap_err();

        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_get_hardware_info_default() {
        let svc = make_service(state_with_mock());
        let resp = svc
            .get_hardware_info(Request::new(GetHardwareInfoRequest {
                backend_name: None,
            }))
            .await
            .unwrap();

        let info = resp.into_inner().info.unwrap();
        assert_eq!(info.name, "test");
    }

    #[tokio::test]
    async fn test_health_check_single_healthy() {
        let svc = make_service(state_with_mock());
        let resp = svc
            .health_check(Request::new(HealthCheckRequest {
                backend_name: Some("test".to_string()),
            }))
            .await
            .unwrap();

        let inner = resp.into_inner();
        assert_eq!(inner.status, ProtoHealthStatus::Healthy as i32);
    }

    #[tokio::test]
    async fn test_health_check_all_healthy() {
        let svc = make_service(state_with_mock());
        let resp = svc
            .health_check(Request::new(HealthCheckRequest {
                backend_name: None,
            }))
            .await
            .unwrap();

        let inner = resp.into_inner();
        assert_eq!(inner.status, ProtoHealthStatus::Healthy as i32);
        assert!(inner.backends.contains_key("test"));
    }

    #[tokio::test]
    async fn test_health_check_mixed_status() {
        let registry = Arc::new(BackendRegistry::default());
        registry.register(MockBackend::simulator("healthy"));
        registry.register(Arc::new(DegradedMockBackend::new("degraded")));
        let state = Arc::new(ServerState::new(registry));

        let svc = make_service(state);
        let resp = svc
            .health_check(Request::new(HealthCheckRequest {
                backend_name: None,
            }))
            .await
            .unwrap();

        let inner = resp.into_inner();
        // Overall should be degraded since one backend is degraded
        assert_eq!(inner.status, ProtoHealthStatus::Degraded as i32);
    }

    #[tokio::test]
    async fn test_health_check_not_found() {
        let svc = make_service(state_with_mock());
        let err = svc
            .health_check(Request::new(HealthCheckRequest {
                backend_name: Some("nope".to_string()),
            }))
            .await
            .unwrap_err();

        assert_eq!(err.code(), tonic::Code::NotFound);
    }

    #[test]
    fn test_sanitize_error_for_status_validation() {
        use crate::error::ValidationError;
        let e = Error::Validation(ValidationError::Field {
            field: "x".into(),
            message: "bad".into(),
        });
        let msg = sanitize_error_for_status(&e);
        assert!(msg.contains("x"));
        assert!(msg.contains("bad"));
    }

    #[test]
    fn test_sanitize_error_for_status_non_validation() {
        let e = Error::Server("internal detail".into());
        let msg = sanitize_error_for_status(&e);
        // In debug mode, should show the detail
        #[cfg(debug_assertions)]
        assert!(msg.contains("internal detail"));
    }
}
