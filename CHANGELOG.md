# Changelog

All notable changes to qubit-os-hardware will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Input validation at API boundary (envelope size, qubit bounds, amplitude limits)
- Timeout protection for Python/QuTiP execution (300s default)
- Rate limiting configuration (enforcement at infrastructure layer)
- SecretString support for IQM auth tokens (secrecy crate)
- Graceful shutdown with configurable timeout (30s default)
- Error message sanitization for production builds
- Comprehensive validation module with DoS protection
- Proto size limits documentation (see qubit-os-proto/LIMITS.md)

### Changed
- Default host binding changed from 0.0.0.0 to 127.0.0.1 (security)
- CORS now restrictive by default (localhost only)

### Removed
- Unused dependencies: anyhow, dotenvy, config, futures

### Security
- Fixed potential infinite hang in QuTiP backend (timeout added)
- Fixed potential memory exhaustion via large envelopes (size limits)
- Fixed potential qubit index out-of-bounds (validation)
- Fixed error messages leaking internal paths/tracebacks

## [0.1.0] - Unreleased

Initial release of QubitOS Hardware Abstraction Layer.

### Features
- gRPC server (tonic) with QuantumBackend service
- REST API facade (axum) at /api/v1/*
- QuTiP backend via PyO3 for simulation
- IQM backend client (optional, behind feature flag)
- Configuration via YAML and environment variables
- Health checks for all backends
- Backend registry with default backend support
