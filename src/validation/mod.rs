// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Input validation for HAL requests.

use crate::config::ResourceLimits;
use crate::error::{Result, ValidationError};

/// Validate pulse execution request parameters.
pub fn validate_execute_pulse_request(
    num_shots: u32,
    pulse_duration_ns: u32,
    num_time_steps: u32,
    limits: &ResourceLimits,
) -> Result<()> {
    if num_shots == 0 {
        return Err(ValidationError::Field {
            field: "num_shots".into(),
            message: "must be greater than 0".into(),
        }
        .into());
    }

    if num_shots > limits.max_shots {
        return Err(ValidationError::ResourceLimit {
            resource: "num_shots".into(),
            limit: limits.max_shots as u64,
            requested: num_shots as u64,
        }
        .into());
    }

    if pulse_duration_ns > limits.max_pulse_duration_ns {
        return Err(ValidationError::ResourceLimit {
            resource: "pulse_duration_ns".into(),
            limit: limits.max_pulse_duration_ns as u64,
            requested: pulse_duration_ns as u64,
        }
        .into());
    }

    if num_time_steps > limits.max_time_steps {
        return Err(ValidationError::ResourceLimit {
            resource: "num_time_steps".into(),
            limit: limits.max_time_steps as u64,
            requested: num_time_steps as u64,
        }
        .into());
    }

    Ok(())
}

/// Validate pulse envelope data.
pub fn validate_pulse_envelope(
    i_envelope: &[f64],
    q_envelope: &[f64],
    num_time_steps: usize,
    max_amplitude: f64,
) -> Result<()> {
    if i_envelope.len() != num_time_steps {
        return Err(ValidationError::Field {
            field: "i_envelope".into(),
            message: format!(
                "length {} does not match num_time_steps {}",
                i_envelope.len(),
                num_time_steps
            ),
        }
        .into());
    }

    if q_envelope.len() != num_time_steps {
        return Err(ValidationError::Field {
            field: "q_envelope".into(),
            message: format!(
                "length {} does not match num_time_steps {}",
                q_envelope.len(),
                num_time_steps
            ),
        }
        .into());
    }

    // Check for NaN or Inf
    for (i, val) in i_envelope.iter().enumerate() {
        if val.is_nan() {
            return Err(ValidationError::Field {
                field: "i_envelope".into(),
                message: format!("contains NaN at index {}", i),
            }
            .into());
        }
        if val.is_infinite() {
            return Err(ValidationError::Field {
                field: "i_envelope".into(),
                message: format!("contains Inf at index {}", i),
            }
            .into());
        }
        if val.abs() > max_amplitude {
            return Err(ValidationError::Field {
                field: "i_envelope".into(),
                message: format!(
                    "amplitude {} at index {} exceeds max {}",
                    val, i, max_amplitude
                ),
            }
            .into());
        }
    }

    for (i, val) in q_envelope.iter().enumerate() {
        if val.is_nan() {
            return Err(ValidationError::Field {
                field: "q_envelope".into(),
                message: format!("contains NaN at index {}", i),
            }
            .into());
        }
        if val.is_infinite() {
            return Err(ValidationError::Field {
                field: "q_envelope".into(),
                message: format!("contains Inf at index {}", i),
            }
            .into());
        }
        if val.abs() > max_amplitude {
            return Err(ValidationError::Field {
                field: "q_envelope".into(),
                message: format!(
                    "amplitude {} at index {} exceeds max {}",
                    val, i, max_amplitude
                ),
            }
            .into());
        }
    }

    Ok(())
}

/// Validate batch request size.
pub fn validate_batch_size(batch_size: usize, limits: &ResourceLimits) -> Result<()> {
    if batch_size == 0 {
        return Err(ValidationError::Field {
            field: "requests".into(),
            message: "batch cannot be empty".into(),
        }
        .into());
    }

    if batch_size > limits.max_batch_size as usize {
        return Err(ValidationError::ResourceLimit {
            resource: "batch_size".into(),
            limit: limits.max_batch_size as u64,
            requested: batch_size as u64,
        }
        .into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_execute_pulse_request() {
        let limits = ResourceLimits::default();

        // Valid request
        assert!(validate_execute_pulse_request(1000, 100, 50, &limits).is_ok());

        // Zero shots
        assert!(validate_execute_pulse_request(0, 100, 50, &limits).is_err());

        // Exceeds max shots
        assert!(validate_execute_pulse_request(1_000_000, 100, 50, &limits).is_err());
    }

    #[test]
    fn test_validate_pulse_envelope() {
        let i_env: Vec<f64> = vec![0.0; 100];
        let q_env: Vec<f64> = vec![0.0; 100];

        // Valid envelope
        assert!(validate_pulse_envelope(&i_env, &q_env, 100, 100.0).is_ok());

        // Wrong length
        assert!(validate_pulse_envelope(&i_env, &q_env, 50, 100.0).is_err());

        // Contains NaN
        let mut bad_env = i_env.clone();
        bad_env[50] = f64::NAN;
        assert!(validate_pulse_envelope(&bad_env, &q_env, 100, 100.0).is_err());
    }
}
