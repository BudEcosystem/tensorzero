# CI Performance Testing

This directory contains scripts for automated performance testing in CI/CD pipelines.

## Scripts

### `run-performance-test.sh`
Main performance test runner that:
- Builds the gateway and mock provider with performance optimizations
- Starts both services
- Runs warmup requests
- Executes load test using Vegeta
- Outputs results in JSON format
- Performs basic threshold checks

**Environment Variables:**
- `PERF_TEST_DURATION`: Test duration (default: 10s)
- `PERF_TEST_RATE`: Requests per second (default: 100)
- `PERF_TEST_TIMEOUT`: Request timeout (default: 2s)
- `PERF_TEST_WARMUP`: Number of warmup requests (default: 50)
- `PERF_TEST_OUTPUT`: Output file path (default: performance-results.json)

### `compare-performance.py`
Compares two performance test results and generates a markdown report:
- Calculates percentage changes for all metrics
- Checks for regressions against configurable thresholds
- Outputs markdown suitable for PR comments
- Exits with error code if regressions are detected

**Usage:**
```bash
python compare-performance.py baseline.json current.json
```

## Integration with GitHub Actions

The performance test job in `.github/workflows/pr.yaml`:
1. Runs on every PR
2. Builds with the `performance` profile (maximum optimizations)
3. Executes the performance test
4. Uploads results as artifacts
5. Downloads baseline from the most recent successful main branch run
6. Compares and posts results as a PR comment
7. Currently set as non-blocking (`continue-on-error: true`)

## Thresholds

Default regression thresholds:
- **P99 Latency**: +10% maximum increase
- **P95 Latency**: +15% maximum increase  
- **Success Rate**: -1% maximum decrease

These can be adjusted in `compare-performance.py`.

## Local Development

To run performance tests locally:

```bash
# Basic run
./ci/performance/run-performance-test.sh

# Custom parameters
PERF_TEST_DURATION=30s PERF_TEST_RATE=500 ./ci/performance/run-performance-test.sh

# Compare with a baseline
./ci/performance/run-performance-test.sh
mv performance-results.json baseline.json
# Make some changes...
./ci/performance/run-performance-test.sh
python ci/performance/compare-performance.py baseline.json performance-results.json
```