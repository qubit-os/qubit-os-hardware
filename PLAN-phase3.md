# Plan: Phase 3.1 — IQM Backend Production-Ready

## Current State

The IQM backend (`src/backend/iqm/mod.rs`, 424 lines, feature-gated `iqm`) already has:
- `IqmBackend` struct with reqwest HTTP client + bearer auth
- `submit_job()` → POST /jobs with circuit JSON
- `wait_for_job()` → GET /jobs/{id} polling (1s interval, timeout-based max)
- `pulse_to_circuit()` — **stub**: only emits `measure` instructions
- `execute_pulse()` — wires submit→poll→convert
- `health_check()` — GET /health
- `get_hardware_info()` — static Garnet info
- 2 tests (config validation only)

## What's Missing (from Roadmap 3.1 exit criteria)

1. **Retry logic with exponential backoff** — Design doc §8.3: max 3 retries with backoff
2. **Real pulse-to-circuit conversion** — Full ZXZ Euler decomposition → PRX+CZ
3. **Proper auth token handling** — `SecretString` from `secrecy` crate (already in deps)
4. **Dynamic hardware info** — Query `/quantum-architecture` endpoint
5. **Calibration set support** — `calibration_set_id` config field
6. **Comprehensive error handling** — Retry transient failures, classify HTTP errors
7. **Tests** — Hand-written test doubles (no mockall), ~20 new tests

## Architecture

### Test Strategy: Hand-written test doubles

Following the Linux kernel / tokio / hyper pattern: **trait-based DI with hand-written fakes**.

- Define `IqmHttpClient` trait with 4 methods
- Production impl: `ReqwestIqmClient` (wraps reqwest + retry logic)
- Test impl: `FakeIqmClient` with configurable responses (canned data, error sequences)
- No mockall, no httpmock — just plain Rust structs

### Gate Decomposition: Full Euler (ZXZ → PRX+CZ)

For single-qubit pulses:
1. Compute rotation angle θ from pulse envelope: `θ = ∫|Ω(t)|dt` (integrated Rabi frequency)
2. Compute phase φ from I/Q ratio: `φ = atan2(Q_avg, I_avg)`
3. Emit `prx(θ, φ)` instruction

For two-qubit pulses:
1. Emit CZ gate between the two target qubits
2. Apply single-qubit PRX corrections on each qubit
3. Measure all targets

Physics: The pulse envelope `I(t) + iQ(t)` drives Rabi oscillations. The integral of the amplitude gives total rotation angle, the I/Q phase gives the rotation axis in the XY plane. This maps directly to IQM's `prx(angle, phase)` native gate.

## Files to Modify (in order)

### 1. `src/config.rs` — Add retry config (~20 lines)

Add to `IqmConfig`:
```rust
pub max_retries: u32,           // default 3
pub retry_base_delay_ms: u64,   // default 500
pub calibration_set_id: Option<String>,  // default None
```
Add default functions, update serde defaults.

### 2. `src/backend/iqm/client.rs` (NEW, ~250 lines)

**`IqmHttpClient` trait:**
```rust
#[async_trait]
pub trait IqmHttpClient: Send + Sync {
    async fn submit_job(&self, request: &IqmJobRequest) -> Result<String, BackendError>;
    async fn get_job_result(&self, job_id: &str) -> Result<IqmJobResult, BackendError>;
    async fn get_quantum_architecture(&self) -> Result<IqmArchitecture, BackendError>;
    async fn health_check(&self) -> Result<bool, BackendError>;
}
```

**`ReqwestIqmClient` impl:**
- Owns `reqwest::Client`, `gateway_url: String`, `auth_token: SecretString`
- `request_with_retry()` — exponential backoff wrapper
  - Retries: 429, 503, 504, reqwest::Error (timeout/connect)
  - No retry: 400, 401, 403, 404
  - Backoff: `min(base_delay * 2^attempt, max_delay)` + jitter
- Each method calls `request_with_retry()` internally
- Auth: `bearer_auth(self.auth_token.expose_secret())`

**New response types:**
```rust
struct IqmArchitecture {
    qubits: Vec<String>,           // ["QB1", "QB2", ...]
    operations: HashMap<String, Vec<Vec<String>>>,  // gate → qubit combos
    name: String,
}
```

### 3. `src/backend/iqm/mod.rs` — Rewrite backend logic (~400 lines, replaces ~280)

**Structural changes:**
- `IqmBackend` becomes generic over `C: IqmHttpClient`
- Constructor takes `IqmHttpClient` impl (production: `ReqwestIqmClient`)
- Remove direct `reqwest::Client` usage — all HTTP goes through client trait

**`pulse_to_circuit()` → full Euler decomposition:**
```rust
fn pulse_to_circuit(&self, request: &ExecutePulseRequest) -> IqmCircuit {
    let mut instructions = Vec::new();
    let num_qubits = request.target_qubits.len();

    if num_qubits == 2 {
        // Two-qubit gate: CZ + single-qubit corrections
        let qb0 = format!("QB{}", request.target_qubits[0] + 1);
        let qb1 = format!("QB{}", request.target_qubits[1] + 1);
        instructions.push(cz_instruction(&qb0, &qb1));
    }

    // Single-qubit rotations from pulse envelope
    for &qubit_idx in &request.target_qubits {
        let qb = format!("QB{}", qubit_idx + 1);
        let (angle, phase) = extract_rotation_params(
            &request.i_envelope, &request.q_envelope
        );
        if angle.abs() > 1e-10 {
            instructions.push(prx_instruction(&qb, angle, phase));
        }
    }

    // Always measure
    for &qubit_idx in &request.target_qubits {
        let qb = format!("QB{}", qubit_idx + 1);
        instructions.push(measure_instruction(&qb));
    }

    IqmCircuit { name: request.pulse_id.clone(), instructions }
}
```

**`extract_rotation_params()`:**
```rust
fn extract_rotation_params(i_envelope: &[f64], q_envelope: &[f64]) -> (f64, f64) {
    // Integrate pulse amplitude: θ = Σ sqrt(I²+Q²) * dt
    let dt = 1.0 / i_envelope.len() as f64;
    let angle: f64 = i_envelope.iter().zip(q_envelope.iter())
        .map(|(i, q)| (i*i + q*q).sqrt() * dt)
        .sum();

    // Average phase: φ = atan2(Σ Q, Σ I)
    let i_sum: f64 = i_envelope.iter().sum();
    let q_sum: f64 = q_envelope.iter().sum();
    let phase = q_sum.atan2(i_sum);

    (angle, phase)
}
```

**`get_hardware_info()` — dynamic with fallback:**
```rust
async fn get_hardware_info(&self) -> Result<HardwareInfo, BackendError> {
    match self.client.get_quantum_architecture().await {
        Ok(arch) => /* convert to HardwareInfo */,
        Err(e) => {
            warn!(error = %e, "Failed to query IQM architecture, using static info");
            Ok(self.static_hardware_info())
        }
    }
}
```

**Result quality detection:**
```rust
let quality = if result.warnings.is_some() {
    ResultQuality::Degraded
} else if total_shots < request.num_shots {
    ResultQuality::PartialFailure
} else {
    ResultQuality::FullSuccess
};
```

### 4. Tests — Hand-written FakeIqmClient (~250 lines in mod.rs tests block)

**`FakeIqmClient` struct:**
```rust
struct FakeIqmClient {
    submit_responses: Mutex<VecDeque<Result<String, BackendError>>>,
    poll_responses: Mutex<VecDeque<Result<IqmJobResult, BackendError>>>,
    arch_response: Mutex<Option<Result<IqmArchitecture, BackendError>>>,
    health_response: Mutex<Result<bool, BackendError>>,
}
```

**Test list (20 tests):**

Retry logic (in client.rs tests):
- `test_retry_on_503` — 2 failures then success
- `test_retry_exhausted` — 4 failures → error
- `test_no_retry_on_401` — immediate auth failure
- `test_no_retry_on_400` — immediate bad request

Job lifecycle:
- `test_submit_job_success`
- `test_submit_job_auth_failure`
- `test_wait_for_job_immediate_ready`
- `test_wait_for_job_pending_then_ready`
- `test_wait_for_job_failed`
- `test_wait_for_job_timeout`

Gate decomposition (pure functions, no mock needed):
- `test_extract_rotation_zero_envelope` — (0, 0) → angle ≈ 0
- `test_extract_rotation_x_gate` — I-only envelope → phase ≈ 0
- `test_extract_rotation_y_gate` — Q-only envelope → phase ≈ π/2
- `test_pulse_to_circuit_single_qubit` — 1 PRX + 1 measure
- `test_pulse_to_circuit_two_qubit` — 1 CZ + 2 PRX + 2 measure
- `test_pulse_to_circuit_zero_amplitude` — measure only

Integration (FakeIqmClient):
- `test_execute_pulse_full_pipeline`
- `test_execute_pulse_qubit_out_of_range`
- `test_health_check_variants` — healthy, degraded, unavailable
- `test_hardware_info_dynamic_fallback`

### 5. Cleanup

- Delete `PLAN-phase3.md` after implementation
- Run `cargo test --features iqm` — all pass
- Run `cargo clippy --features iqm` — clean

## Estimated Impact

- New/changed lines: ~700
- New tests: ~20 (feature-gated behind `iqm`)
- Total IQM test count: 2 → 22
- Covers: auth, retry, decomposition, result parsing, error paths
