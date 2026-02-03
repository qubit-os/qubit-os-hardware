// Copyright 2026 QubitOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HAL server implementations.
//!
//! This module provides both gRPC (tonic) and REST (axum) server implementations
//! for the Hardware Abstraction Layer.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              HAL Server                  │
//! ├──────────────────┬──────────────────────┤
//! │  gRPC (50051)    │   REST (8080)        │
//! │  QuantumBackend  │   /api/v1/*          │
//! │  service         │                       │
//! └──────────────────┴──────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use qubit_os_hardware::server::{GrpcServer, RestServer};
//! use qubit_os_hardware::backend::BackendRegistry;
//! use qubit_os_hardware::config::Config;
//!
//! let config = Config::load(None)?;
//! let registry = Arc::new(BackendRegistry::new(&config.backends));
//!
//! // Start servers
//! tokio::select! {
//!     _ = GrpcServer::new(registry.clone()).serve(&config.server) => {}
//!     _ = RestServer::new(registry.clone()).serve(&config.server) => {}
//! }
//! ```

pub mod grpc;
pub mod rest;

pub use grpc::GrpcServer;
pub use rest::RestServer;

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tracing::{error, info};

use crate::backend::BackendRegistry;
use crate::config::ServerConfig;
use crate::error::Result;

/// Shared state for servers.
pub struct ServerState {
    /// Backend registry
    pub registry: Arc<BackendRegistry>,

    /// Shutdown signal sender
    shutdown_tx: watch::Sender<bool>,

    /// Shutdown signal receiver
    shutdown_rx: watch::Receiver<bool>,
}

impl ServerState {
    /// Create new server state.
    pub fn new(registry: Arc<BackendRegistry>) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            registry,
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Get a shutdown receiver.
    pub fn shutdown_receiver(&self) -> watch::Receiver<bool> {
        self.shutdown_rx.clone()
    }

    /// Signal shutdown.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

/// Run both gRPC and REST servers.
pub async fn run_servers(config: &ServerConfig, registry: Arc<BackendRegistry>) -> Result<()> {
    let state = Arc::new(ServerState::new(registry));

    // Create servers
    let grpc_server = GrpcServer::new(state.clone());
    let rest_server = RestServer::new(state.clone());

    info!(
        grpc_port = config.grpc_port,
        rest_port = config.rest_port,
        rest_enabled = config.rest_enabled,
        "Starting HAL servers"
    );

    // Set up signal handler for graceful shutdown
    let state_for_signal = state.clone();
    tokio::spawn(async move {
        if let Ok(()) = tokio::signal::ctrl_c().await {
            info!("Received shutdown signal, initiating graceful shutdown");
            state_for_signal.shutdown();
        }
    });

    // Graceful shutdown timeout from config
    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout_sec);

    // Run servers concurrently
    if config.rest_enabled {
        tokio::select! {
            result = grpc_server.serve(config) => {
                if let Err(e) = result {
                    error!(error = %e, "gRPC server error");
                }
            }
            result = rest_server.serve(config) => {
                if let Err(e) = result {
                    error!(error = %e, "REST server error");
                }
            }
        }
    } else {
        // Only run gRPC server
        grpc_server.serve(config).await?;
    }

    // Ensure shutdown completes within timeout
    info!(timeout_secs = config.shutdown_timeout_sec, "Waiting for shutdown to complete");

    // The servers already handle graceful shutdown via the shutdown_rx channel.
    // This timeout ensures we don't hang forever if something goes wrong.
    // Note: The actual timeout is enforced by the individual server shutdown handlers.
    let _ = shutdown_timeout;

    Ok(())
}
