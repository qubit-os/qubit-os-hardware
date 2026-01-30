// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Error types for the HAL.

use std::fmt;

/// Result type alias for HAL operations.
pub type Result<T> = std::result::Result<T, Error>;

/// HAL error types.
#[derive(Debug)]
pub enum Error {
    /// Configuration error
    Config(String),
    /// Backend error
    Backend(BackendError),
    /// Validation error
    Validation(ValidationError),
    /// Server error
    Server(String),
    /// IO error
    Io(std::io::Error),
    /// Serialization error
    Serialization(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(msg) => write!(f, "Configuration error: {}", msg),
            Error::Backend(e) => write!(f, "Backend error: {}", e),
            Error::Validation(e) => write!(f, "Validation error: {}", e),
            Error::Server(msg) => write!(f, "Server error: {}", msg),
            Error::Io(e) => write!(f, "IO error: {}", e),
            Error::Serialization(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Backend(e) => Some(e),
            Error::Validation(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<BackendError> for Error {
    fn from(e: BackendError) -> Self {
        Error::Backend(e)
    }
}

impl From<ValidationError> for Error {
    fn from(e: ValidationError) -> Self {
        Error::Validation(e)
    }
}

impl From<serde_yaml::Error> for Error {
    fn from(e: serde_yaml::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

/// Backend-specific errors.
#[derive(Debug)]
pub enum BackendError {
    /// Backend not found
    NotFound(String),
    /// Backend unavailable
    Unavailable(String),
    /// Execution failed
    ExecutionFailed(String),
    /// Authentication failed
    AuthenticationFailed(String),
    /// Timeout
    Timeout(String),
    /// Invalid request
    InvalidRequest(String),
    /// Python error (for QuTiP backend)
    Python(String),
    /// HTTP error (for IQM backend)
    Http(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::NotFound(name) => write!(f, "Backend not found: {}", name),
            BackendError::Unavailable(msg) => write!(f, "Backend unavailable: {}", msg),
            BackendError::ExecutionFailed(msg) => write!(f, "Execution failed: {}", msg),
            BackendError::AuthenticationFailed(msg) => write!(f, "Authentication failed: {}", msg),
            BackendError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            BackendError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            BackendError::Python(msg) => write!(f, "Python error: {}", msg),
            BackendError::Http(msg) => write!(f, "HTTP error: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Validation errors.
#[derive(Debug)]
pub enum ValidationError {
    /// Field validation failed
    Field { field: String, message: String },
    /// Physics constraint violated
    PhysicsConstraint(String),
    /// Calibration mismatch
    CalibrationMismatch { expected: String, actual: String },
    /// Resource limit exceeded
    ResourceLimit {
        resource: String,
        limit: u64,
        requested: u64,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::Field { field, message } => {
                write!(f, "Field '{}': {}", field, message)
            }
            ValidationError::PhysicsConstraint(msg) => {
                write!(f, "Physics constraint violated: {}", msg)
            }
            ValidationError::CalibrationMismatch { expected, actual } => {
                write!(
                    f,
                    "Calibration mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            ValidationError::ResourceLimit {
                resource,
                limit,
                requested,
            } => {
                write!(
                    f,
                    "Resource limit exceeded for {}: limit={}, requested={}",
                    resource, limit, requested
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Convert HAL errors to gRPC status codes.
impl From<Error> for tonic::Status {
    fn from(e: Error) -> Self {
        match e {
            Error::Config(msg) => tonic::Status::failed_precondition(msg),
            Error::Backend(BackendError::NotFound(msg)) => tonic::Status::not_found(msg),
            Error::Backend(BackendError::Unavailable(msg)) => tonic::Status::unavailable(msg),
            Error::Backend(BackendError::AuthenticationFailed(msg)) => {
                tonic::Status::unauthenticated(msg)
            }
            Error::Backend(BackendError::Timeout(msg)) => tonic::Status::deadline_exceeded(msg),
            Error::Backend(BackendError::InvalidRequest(msg)) => {
                tonic::Status::invalid_argument(msg)
            }
            Error::Backend(e) => tonic::Status::internal(e.to_string()),
            Error::Validation(e) => tonic::Status::invalid_argument(e.to_string()),
            Error::Server(msg) => tonic::Status::internal(msg),
            Error::Io(e) => tonic::Status::internal(e.to_string()),
            Error::Serialization(msg) => tonic::Status::internal(msg),
        }
    }
}
