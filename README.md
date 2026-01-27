# QubitOS Hardware Abstraction Layer (HAL)

[![CI](https://github.com/qubit-os/qubit-os-hardware/actions/workflows/ci.yaml/badge.svg)](https://github.com/qubit-os/qubit-os-hardware/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Rust implementation of the QubitOS Hardware Abstraction Layer - the bridge between pulse optimization and quantum backends.

## Overview

The HAL provides:

- **gRPC Server** - Protocol buffer interface for pulse execution
- **REST API** - HTTP/JSON facade for the gRPC service
- **Backend Registry** - Pluggable backend system for simulators and hardware
- **QuTiP Backend** - Local quantum simulator (via PyO3)
- **IQM Backend** - Cloud access to IQM quantum processors

## Quick Start

### Running the HAL

```bash
# Development mode
cargo run -- --config config.yaml

# Or with environment variables
QUBITOS_LOG_LEVEL=debug cargo run

# Production (release build)
cargo build --release
./target/release/qubit-os-hal --config config.yaml
```

### Docker

```bash
# Build
docker build -t qubit-os-hardware .

# Run
docker run -p 50051:50051 -p 8080:8080 qubit-os-hardware
```

### Configuration

```yaml
# config.yaml
server:
  grpc_port: 50051
  rest_port: 8080
  host: "0.0.0.0"

backends:
  qutip_simulator:
    enabled: true
    default: true
  iqm_garnet:
    enabled: false
    gateway_url: "${IQM_GATEWAY_URL}"
    auth_token: "${IQM_AUTH_TOKEN}"

logging:
  level: "info"
  format: "json"
```

## Architecture

```
┌─────────────────────────────────────────┐
│              HAL Server                  │
├──────────────────┬──────────────────────┤
│   gRPC Service   │   REST Service       │
│   (tonic)        │   (axum)             │
├──────────────────┴──────────────────────┤
│           Backend Registry               │
├────────────────┬────────────────────────┤
│ QuTiP Backend  │     IQM Backend        │
│ (PyO3)         │     (reqwest)          │
└────────────────┴────────────────────────┘
```

## Development

### Prerequisites

- Rust 1.75+
- Python 3.11+ (for QuTiP backend)
- Protocol Buffers compiler (protoc)

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with logging
RUST_LOG=debug cargo run
```

### Code Structure

```
src/
├── main.rs           # Entry point
├── lib.rs            # Library exports
├── config.rs         # Configuration handling
├── server/
│   ├── mod.rs
│   ├── grpc.rs       # gRPC service implementation
│   └── rest.rs       # REST API facade
├── backend/
│   ├── mod.rs
│   ├── trait.rs      # QuantumBackend trait
│   ├── registry.rs   # Backend registration
│   ├── qutip/        # QuTiP simulator backend
│   └── iqm/          # IQM hardware backend
├── validation/
│   ├── mod.rs
│   ├── pulse.rs      # Pulse validation
│   └── hamiltonian.rs
└── error.rs          # Error types
```

## Adding a New Backend

1. Implement the `QuantumBackend` trait:

```rust
#[async_trait]
impl QuantumBackend for MyBackend {
    fn name(&self) -> &str { "my_backend" }
    fn backend_type(&self) -> BackendType { BackendType::Hardware }
    
    async fn execute_pulse(&self, request: ExecutePulseRequest) 
        -> Result<MeasurementResult, BackendError> {
        // Implementation
    }
    
    // ... other methods
}
```

2. Register in `backend/mod.rs`

3. Add configuration in `config.rs`

4. Write tests in `tests/`

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
