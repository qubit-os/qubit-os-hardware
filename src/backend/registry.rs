// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Backend registry for managing quantum backends.
//!
//! The registry provides discovery, registration, and lookup of quantum backends.
//! It supports both static (compile-time) and dynamic (runtime) backend registration.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use super::{BackendType, QuantumBackend};
use crate::config::{BackendsConfig, ResourceLimits};
use crate::error::{BackendError, Error, Result};

/// Backend registry for managing quantum backends.
///
/// The registry is thread-safe and can be shared across async tasks.
///
/// # Example
///
/// ```ignore
/// use qubit_os_hardware::backend::{BackendRegistry, qutip::QutipBackend};
/// use qubit_os_hardware::config::Config;
///
/// let config = Config::default();
/// let mut registry = BackendRegistry::new(&config.backends);
///
/// // Register backends
/// registry.register(Box::new(QutipBackend::new(&config.backends.qutip_simulator)?));
///
/// // Get a backend
/// let backend = registry.get("qutip_simulator")?;
/// ```
pub struct BackendRegistry {
    /// Registered backends
    backends: RwLock<HashMap<String, Arc<dyn QuantumBackend>>>,

    /// Default backend name
    default_backend: RwLock<Option<String>>,

    /// Global resource limits
    limits: ResourceLimits,
}

impl BackendRegistry {
    /// Create a new backend registry.
    pub fn new(_config: &BackendsConfig) -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
            default_backend: RwLock::new(None),
            limits: ResourceLimits::default(),
        }
    }

    /// Create a registry with specific resource limits.
    pub fn with_limits(limits: ResourceLimits) -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
            default_backend: RwLock::new(None),
            limits,
        }
    }

    /// Register a backend.
    ///
    /// If a backend with the same name already exists, it will be replaced.
    pub fn register(&self, backend: Arc<dyn QuantumBackend>) {
        let name = backend.name().to_string();
        info!(backend = %name, "Registering backend");

        let mut backends = self.backends.write();
        backends.insert(name.clone(), backend);

        // If this is the first backend, make it the default
        let mut default = self.default_backend.write();
        if default.is_none() {
            debug!(backend = %name, "Setting as default backend");
            *default = Some(name);
        }
    }

    /// Set the default backend.
    pub fn set_default(&self, name: &str) -> Result<()> {
        let backends = self.backends.read();
        if !backends.contains_key(name) {
            return Err(Error::Backend(BackendError::NotFound(name.to_string())));
        }

        let mut default = self.default_backend.write();
        *default = Some(name.to_string());
        info!(backend = %name, "Set as default backend");
        Ok(())
    }

    /// Get a backend by name.
    pub fn get(&self, name: &str) -> Result<Arc<dyn QuantumBackend>> {
        let backends = self.backends.read();
        backends
            .get(name)
            .cloned()
            .ok_or_else(|| Error::Backend(BackendError::NotFound(name.to_string())))
    }

    /// Get the default backend.
    pub fn get_default(&self) -> Result<Arc<dyn QuantumBackend>> {
        let default = self.default_backend.read();
        match default.as_ref() {
            Some(name) => self.get(name),
            None => Err(Error::Backend(BackendError::NotFound(
                "No default backend configured".to_string(),
            ))),
        }
    }

    /// Get a backend by name, or the default if name is None.
    pub fn get_or_default(&self, name: Option<&str>) -> Result<Arc<dyn QuantumBackend>> {
        match name {
            Some(n) => self.get(n),
            None => self.get_default(),
        }
    }

    /// List all registered backend names.
    pub fn list(&self) -> Vec<String> {
        let backends = self.backends.read();
        backends.keys().cloned().collect()
    }

    /// List all backends with their types.
    pub fn list_with_types(&self) -> Vec<(String, BackendType)> {
        let backends = self.backends.read();
        backends
            .iter()
            .map(|(name, backend)| (name.clone(), backend.backend_type()))
            .collect()
    }

    /// Check if a backend is registered.
    pub fn contains(&self, name: &str) -> bool {
        let backends = self.backends.read();
        backends.contains_key(name)
    }

    /// Get the number of registered backends.
    pub fn len(&self) -> usize {
        let backends = self.backends.read();
        backends.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unregister a backend.
    pub fn unregister(&self, name: &str) -> Option<Arc<dyn QuantumBackend>> {
        let mut backends = self.backends.write();
        let removed = backends.remove(name);

        if removed.is_some() {
            info!(backend = %name, "Unregistered backend");

            // If this was the default, clear it
            let mut default = self.default_backend.write();
            if default.as_ref() == Some(&name.to_string()) {
                warn!(backend = %name, "Unregistered default backend");
                *default = None;
            }
        }

        removed
    }

    /// Get the global resource limits.
    pub fn limits(&self) -> &ResourceLimits {
        &self.limits
    }

    /// Get the default backend name.
    pub fn default_backend_name(&self) -> Option<String> {
        self.default_backend.read().clone()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self {
            backends: RwLock::new(HashMap::new()),
            default_backend: RwLock::new(None),
            limits: ResourceLimits::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::r#trait::{
        ExecutePulseRequest, HardwareInfo, HealthStatus, MeasurementResult, ResultQuality,
    };
    use async_trait::async_trait;
    use std::collections::HashMap;

    /// Mock backend for testing
    struct MockBackend {
        name: String,
        backend_type: BackendType,
        limits: ResourceLimits,
    }

    impl MockBackend {
        fn new(name: &str, backend_type: BackendType) -> Self {
            Self {
                name: name.to_string(),
                backend_type,
                limits: ResourceLimits::default(),
            }
        }
    }

    #[async_trait]
    impl QuantumBackend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }

        fn backend_type(&self) -> BackendType {
            self.backend_type
        }

        async fn execute_pulse(
            &self,
            _request: ExecutePulseRequest,
        ) -> std::result::Result<MeasurementResult, BackendError> {
            Ok(MeasurementResult {
                bitstring_counts: HashMap::new(),
                total_shots: 1000,
                successful_shots: 1000,
                quality: ResultQuality::FullSuccess,
                fidelity_estimate: Some(0.99),
                state_vector: None,
            })
        }

        async fn get_hardware_info(&self) -> std::result::Result<HardwareInfo, BackendError> {
            Ok(HardwareInfo {
                name: self.name.clone(),
                backend_type: self.backend_type,
                tier: "local".to_string(),
                num_qubits: 2,
                available_qubits: vec![0, 1],
                supported_gates: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
                supports_state_vector: true,
                supports_noise_model: false,
                software_version: "1.0.0".to_string(),
                limits: self.limits.clone(),
            })
        }

        async fn health_check(&self) -> std::result::Result<HealthStatus, BackendError> {
            Ok(HealthStatus::Healthy)
        }

        fn resource_limits(&self) -> &ResourceLimits {
            &self.limits
        }
    }

    #[test]
    fn test_registry_register_and_get() {
        let registry = BackendRegistry::default();
        let backend = Arc::new(MockBackend::new("test", BackendType::Simulator));

        registry.register(backend);

        assert!(registry.contains("test"));
        assert!(!registry.is_empty());
        assert_eq!(registry.len(), 1);

        let retrieved = registry.get("test").unwrap();
        assert_eq!(retrieved.name(), "test");
    }

    #[test]
    fn test_registry_default_backend() {
        let registry = BackendRegistry::default();
        let backend1 = Arc::new(MockBackend::new("first", BackendType::Simulator));
        let backend2 = Arc::new(MockBackend::new("second", BackendType::Hardware));

        // First registered becomes default
        registry.register(backend1);
        assert_eq!(registry.default_backend_name(), Some("first".to_string()));

        // Second doesn't change default
        registry.register(backend2);
        assert_eq!(registry.default_backend_name(), Some("first".to_string()));

        // Explicit set_default
        registry.set_default("second").unwrap();
        assert_eq!(registry.default_backend_name(), Some("second".to_string()));
    }

    #[test]
    fn test_registry_get_or_default() {
        let registry = BackendRegistry::default();
        let backend = Arc::new(MockBackend::new("test", BackendType::Simulator));
        registry.register(backend);

        // With name
        let result = registry.get_or_default(Some("test"));
        assert!(result.is_ok());

        // Without name (uses default)
        let result = registry.get_or_default(None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_unregister() {
        let registry = BackendRegistry::default();
        let backend = Arc::new(MockBackend::new("test", BackendType::Simulator));
        registry.register(backend);

        assert!(registry.contains("test"));

        let removed = registry.unregister("test");
        assert!(removed.is_some());
        assert!(!registry.contains("test"));
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_list() {
        let registry = BackendRegistry::default();
        registry.register(Arc::new(MockBackend::new("sim", BackendType::Simulator)));
        registry.register(Arc::new(MockBackend::new("hw", BackendType::Hardware)));

        let names = registry.list();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"sim".to_string()));
        assert!(names.contains(&"hw".to_string()));

        let with_types = registry.list_with_types();
        assert_eq!(with_types.len(), 2);
    }
}
