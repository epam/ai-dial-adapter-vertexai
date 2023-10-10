import os


def pytest_configure(config):
    """
    Setting up fake environment variables for unit tests.
    """
    os.environ["DEFAULT_REGION"] = "dummy_region"
    os.environ["GCP_PROJECT_ID"] = "dummy_project_id"
