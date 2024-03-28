import os


def pytest_configure(config):
    """
    Setting up fake environment variables for unit tests.
    """
    os.environ["DEFAULT_REGION"] = "us-central1"
    os.environ["GCP_PROJECT_ID"] = "dummy_project_id"
