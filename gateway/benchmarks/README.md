# Benchmarks

## CI Performance Testing

Performance tests run automatically on every PR to detect regressions. The CI system:

1. **Runs lightweight performance tests** (10 seconds, 100 RPS) on every PR
2. **Compares results against baseline** from the main branch
3. **Posts results as a PR comment** with detailed metrics
4. **Fails if regressions exceed thresholds**:
   - P99 latency: +10% maximum increase allowed
   - P95 latency: +15% maximum increase allowed
   - Success rate: -1% maximum decrease allowed

### Running Performance Tests Locally

```bash
# Run the CI performance test
./ci/performance/run-performance-test.sh

# Run with custom parameters
PERF_TEST_DURATION=30s PERF_TEST_RATE=500 ./ci/performance/run-performance-test.sh

# Compare two results
python ci/performance/compare-performance.py baseline.json current.json
```

### Performance Test Configuration

The CI tests use:
- **Mock inference provider**: Simulates OpenAI API with consistent 10ms response time
- **No observability**: Tests raw gateway performance without telemetry overhead
- **Warmup phase**: 50 requests before measurement begins
- **Vegeta load tester**: Industry-standard HTTP load testing tool

## Full Benchmarks

## TensorZero Gateway vs. LiteLLM Proxy (LiteLLM Gateway)

### Environment Setup

- Launch an AWS EC2 Instance: `c7i.xlarge` (4 vCPUs, 8 GB RAM)
- Increase the limits for open file descriptors:

  - Run `sudo vim /etc/security/limits.conf` and add the following lines:
    ```
    *               soft    nofile          65536
    *               hard    nofile          65536
    ```
  - Run `sudo vim /etc/pam.d/common-session` and add the following line:
    ```
    session required pam_limits.so
    ```
  - Reboot the instance with `sudo reboot`
  - Run `ulimit -Hn` and `ulimit -Sn` to check that the limits are now `65536`

- Install Python 3.10.14.
- Install LiteLLM: `pip install 'litellm[proxy]'==1.34.42`
- Install Rust 1.80.1.
- Install `vegeta` [→](https://github.com/tsenart/vegeta).
- Set the `OPENAI_API_KEY` environment variable to anything (e.g. `OPENAI_API_KEY=test`).

### Test Setup

- Launch the mock inference provider in performance mode:

  ```bash
  cargo run --profile performance --bin mock-inference-provider
  ```

#### TensorZero Gateway

- Launch the TensorZero Gateway in performance mode (without observability):

  ```bash
  cargo run --profile performance --bin gateway tensorzero-internal/tests/load/tensorzero-without-observability.toml
  ```

- Run the benchmark:
  ```bash
  sh tensorzero-internal/tests/load/simple/run.sh
  ```

#### LiteLLM Gateway (LiteLLM Proxy)

- Launch the LiteLLM Gateway:

  ```
  litellm --config tensorzero-internal/tests/load/simple-litellm/config.yaml --num_workers=4
  ```

- Run the benchmark:

  ```bash
  sh tensorzero-internal/tests/load/simple-litellm/run.sh
  ```
