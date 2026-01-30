// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! IQM Garnet hardware backend.
//!
//! This backend provides integration with IQM's quantum hardware through their
//! REST API. The IQM Garnet system is a 20-qubit superconducting quantum computer.
//!
//! # Requirements
//!
//! - IQM Gateway URL (via config or IQM_GATEWAY_URL env var)
//! - Authentication token (via config or IQM_AUTH_TOKEN env var)
//!
//! # Usage
//!
//! ```ignore
//! use qubit_os_hardware::backend::iqm::IqmBackend;
//! use qubit_os_hardware::config::IqmConfig;
//!
//! let config = IqmConfig {
//!     enabled: true,
//!     gateway_url: Some("https://garnet.iqm.fi/api/v1".to_string()),
//!     auth_token: Some("your-token".to_string()),
//!     timeout_sec: 30,
//! };
//!
//! let backend = IqmBackend::new(&config)?;
//! ```

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, error, info, warn};

use super::{BackendType, QuantumBackend};
use crate::config::{IqmConfig, ResourceLimits};
use crate::error::BackendError;

use super::r#trait::{
    ExecutePulseRequest, HardwareInfo, HealthStatus, MeasurementResult, ResultQuality,
};

/// IQM API job request.
#[derive(Debug, Serialize)]
struct IqmJobRequest {
    /// Pulse sequences
    circuits: Vec<IqmCircuit>,
    /// Number of shots
    shots: u32,
    /// Calibration set ID (optional)
    calibration_set_id: Option<String>,
}

/// IQM circuit representation.
#[derive(Debug, Serialize)]
struct IqmCircuit {
    /// Circuit name
    name: String,
    /// Instructions
    instructions: Vec<IqmInstruction>,
}

/// IQM instruction.
#[derive(Debug, Serialize)]
struct IqmInstruction {
    /// Instruction name
    name: String,
    /// Qubit indices
    qubits: Vec<String>,
    /// Arguments (e.g., angles)
    args: HashMap<String, f64>,
}

/// IQM job response.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct IqmJobResponse {
    /// Job ID
    id: String,
    /// Status
    status: String,
}

/// IQM job result.
#[derive(Debug, Deserialize)]
struct IqmJobResult {
    /// Job status
    status: String,
    /// Measurement results
    measurements: Option<Vec<IqmMeasurement>>,
    /// Error message (if failed)
    message: Option<String>,
}

/// IQM measurement result.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct IqmMeasurement {
    /// Circuit index
    circuit_index: usize,
    /// Measurement outcomes (list of bitstrings as integers)
    result: Vec<Vec<i32>>,
}

/// IQM Garnet hardware backend.
pub struct IqmBackend {
    /// Backend name
    name: String,

    /// HTTP client
    client: Client,

    /// Gateway URL
    gateway_url: String,

    /// Authentication token
    auth_token: String,

    /// Resource limits
    limits: ResourceLimits,

    /// Request timeout
    timeout: Duration,
}

impl IqmBackend {
    /// Create a new IQM backend.
    pub fn new(config: &IqmConfig) -> Result<Self, BackendError> {
        let gateway_url = config.gateway_url.clone().ok_or_else(|| {
            BackendError::InvalidRequest(
                "IQM gateway URL not configured. Set IQM_GATEWAY_URL or config.backends.iqm_garnet.gateway_url".to_string()
            )
        })?;

        let auth_token = config.auth_token.clone().ok_or_else(|| {
            BackendError::AuthenticationFailed(
                "IQM auth token not configured. Set IQM_AUTH_TOKEN or config.backends.iqm_garnet.auth_token".to_string()
            )
        })?;

        let timeout = Duration::from_secs(config.timeout_sec);

        let client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| BackendError::Http(format!("Failed to create HTTP client: {}", e)))?;

        info!(gateway = %gateway_url, "Initializing IQM backend");

        Ok(Self {
            name: "iqm_garnet".to_string(),
            client,
            gateway_url,
            auth_token,
            limits: ResourceLimits {
                max_qubits: 20, // IQM Garnet has 20 qubits
                max_shots: 100_000,
                ..Default::default()
            },
            timeout,
        })
    }

    /// Submit a job to IQM.
    async fn submit_job(&self, request: &IqmJobRequest) -> Result<String, BackendError> {
        let url = format!("{}/jobs", self.gateway_url);

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.auth_token)
            .json(request)
            .send()
            .await
            .map_err(|e| BackendError::Http(format!("Failed to submit job: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(BackendError::Http(format!(
                "Job submission failed ({}): {}",
                status, body
            )));
        }

        let job_response: IqmJobResponse = response
            .json()
            .await
            .map_err(|e| BackendError::Http(format!("Failed to parse job response: {}", e)))?;

        debug!(job_id = %job_response.id, "Job submitted to IQM");
        Ok(job_response.id)
    }

    /// Poll for job completion.
    async fn wait_for_job(&self, job_id: &str) -> Result<IqmJobResult, BackendError> {
        let url = format!("{}/jobs/{}", self.gateway_url, job_id);
        let poll_interval = Duration::from_secs(1);
        let max_polls = (self.timeout.as_secs() / poll_interval.as_secs()) as usize;

        for i in 0..max_polls {
            let response = self
                .client
                .get(&url)
                .bearer_auth(&self.auth_token)
                .send()
                .await
                .map_err(|e| BackendError::Http(format!("Failed to poll job: {}", e)))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(BackendError::Http(format!(
                    "Job poll failed ({}): {}",
                    status, body
                )));
            }

            let result: IqmJobResult = response
                .json()
                .await
                .map_err(|e| BackendError::Http(format!("Failed to parse job result: {}", e)))?;

            match result.status.as_str() {
                "ready" => {
                    debug!(job_id = %job_id, "Job completed");
                    return Ok(result);
                }
                "failed" => {
                    let msg = result
                        .message
                        .unwrap_or_else(|| "Unknown error".to_string());
                    error!(job_id = %job_id, error = %msg, "Job failed");
                    return Err(BackendError::ExecutionFailed(msg));
                }
                "pending" | "running" => {
                    debug!(job_id = %job_id, poll = i, "Job still running");
                    tokio::time::sleep(poll_interval).await;
                }
                status => {
                    warn!(job_id = %job_id, status = %status, "Unknown job status");
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }

        Err(BackendError::Timeout(format!(
            "Job {} did not complete within timeout",
            job_id
        )))
    }

    /// Convert pulse request to IQM circuit.
    ///
    /// Note: This is a simplified conversion. Real implementation would need to
    /// decompose the pulse into IQM's native gate set.
    fn pulse_to_circuit(&self, request: &ExecutePulseRequest) -> IqmCircuit {
        // For now, we create a simple identity circuit
        // Real implementation would convert pulse envelopes to IQM gates
        let instructions = request
            .target_qubits
            .iter()
            .map(|&q| {
                IqmInstruction {
                    name: "measure".to_string(),
                    qubits: vec![format!("QB{}", q + 1)], // IQM uses 1-indexed qubits
                    args: HashMap::new(),
                }
            })
            .collect();

        IqmCircuit {
            name: request.pulse_id.clone(),
            instructions,
        }
    }
}

#[async_trait]
impl QuantumBackend for IqmBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Hardware
    }

    async fn execute_pulse(
        &self,
        request: ExecutePulseRequest,
    ) -> Result<MeasurementResult, BackendError> {
        debug!(
            pulse_id = %request.pulse_id,
            num_shots = request.num_shots,
            "Executing pulse on IQM backend"
        );

        // Validate request
        if request.target_qubits.iter().any(|&q| q >= 20) {
            return Err(BackendError::InvalidRequest(
                "Target qubit exceeds IQM Garnet's 20 qubits".to_string(),
            ));
        }

        // Convert to IQM circuit
        let circuit = self.pulse_to_circuit(&request);

        // Build job request
        let job_request = IqmJobRequest {
            circuits: vec![circuit],
            shots: request.num_shots,
            calibration_set_id: None,
        };

        // Submit and wait for results
        let job_id = self.submit_job(&job_request).await?;
        let result = self.wait_for_job(&job_id).await?;

        // Convert IQM results to our format
        let measurements = result.measurements.ok_or_else(|| {
            BackendError::ExecutionFailed("No measurements in result".to_string())
        })?;

        let mut bitstring_counts: HashMap<String, u32> = HashMap::new();
        let num_qubits = request.target_qubits.len();

        for measurement in &measurements {
            for shot_result in &measurement.result {
                // Convert bit array to bitstring
                let bitstring: String = shot_result
                    .iter()
                    .take(num_qubits)
                    .map(|&b| if b == 0 { '0' } else { '1' })
                    .collect();
                *bitstring_counts.entry(bitstring).or_insert(0) += 1;
            }
        }

        let total_shots = bitstring_counts.values().sum();

        Ok(MeasurementResult {
            bitstring_counts,
            total_shots,
            successful_shots: total_shots,
            quality: ResultQuality::FullSuccess,
            fidelity_estimate: None,
            state_vector: None, // Hardware doesn't provide state vectors
        })
    }

    async fn get_hardware_info(&self) -> Result<HardwareInfo, BackendError> {
        // Could query IQM API for actual hardware info
        // For now, return static info about Garnet
        Ok(HardwareInfo {
            name: self.name.clone(),
            backend_type: BackendType::Hardware,
            tier: "cloud".to_string(),
            num_qubits: 20,
            available_qubits: (0..20).collect(),
            supported_gates: vec![
                "prx".to_string(), // Phased RX
                "cz".to_string(),  // CZ gate
                "measure".to_string(),
            ],
            supports_state_vector: false,
            supports_noise_model: false, // Real hardware has real noise
            software_version: "IQM Garnet".to_string(),
            limits: self.limits.clone(),
        })
    }

    async fn health_check(&self) -> Result<HealthStatus, BackendError> {
        let url = format!("{}/health", self.gateway_url);

        match self.client.get(&url).send().await {
            Ok(response) if response.status().is_success() => Ok(HealthStatus::Healthy),
            Ok(response) => {
                warn!(status = %response.status(), "IQM health check returned non-success");
                Ok(HealthStatus::Degraded)
            }
            Err(e) => {
                error!(error = %e, "IQM health check failed");
                Ok(HealthStatus::Unavailable)
            }
        }
    }

    fn resource_limits(&self) -> &ResourceLimits {
        &self.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iqm_backend_requires_config() {
        let config = IqmConfig::default();
        let result = IqmBackend::new(&config);
        assert!(result.is_err()); // Should fail without gateway URL
    }

    #[test]
    fn test_iqm_backend_with_config() {
        let config = IqmConfig {
            enabled: true,
            gateway_url: Some("https://example.iqm.fi/api".to_string()),
            auth_token: Some("test-token".to_string()),
            timeout_sec: 30,
        };

        let result = IqmBackend::new(&config);
        assert!(result.is_ok());

        let backend = result.unwrap();
        assert_eq!(backend.name(), "iqm_garnet");
        assert_eq!(backend.backend_type(), BackendType::Hardware);
    }
}
