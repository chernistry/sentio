# Sentio RAG Testing

This directory contains tests for the Sentio RAG project, including unit tests, integration tests, and performance tests.

## Test Structure

- `test_api.py` - API endpoint unit tests
- `test_azure_integration.py` - Azure service integration tests
- `locustfile.py` - Load tests using Locust

## Running Tests

### Unit Tests

To run unit tests:

```bash
cd root
pytest tests/test_api.py -v
```

### Azure Integration Tests

To run Azure integration tests, configure environment variables in `.env.azure` file, then run:

```bash
cd root
pytest tests/test_azure_integration.py -v
```

Note: Integration tests require access to Azure resources. If environment variables are not configured, tests will be skipped.

### Load Tests

To run load tests with Locust:

```bash
cd root
locust -f tests/locustfile.py --host=https://your-api-url
```

Then open http://localhost:8089 in your browser to access the Locust web interface.

## Load Testing Parameters

- **Users**: Start with 10 users and gradually increase to 50-100
- **Rate**: Start with 1 user per second and increase as needed
- **Run Time**: Recommended to run tests for 5-10 minutes for stable results
- **Metrics**: Monitor response time (should be <2s for P95) and error rate (should be <1%)

## Adding New Tests

When adding new tests, follow these guidelines:

1. Use `pytest` for unit and integration tests
2. Follow `test_*.py` naming convention for test files
3. Prefix each test function with `test_`
4. Use fixtures for common setup code
5. Add documentation explaining test purpose

## Continuous Integration

Tests automatically run in GitHub Actions on each push to the repository. Configuration is in `.github/workflows/tests.yml`. 