"""
QPIAI Quantum SDK Configuration Module

This module provides centralized configuration management for the QPIAI Quantum SDK,
including environment variable loading and default values.

Environment Variables:
    QPIAI_SERVER_URL: Base URL for the QPIAI Quantum server (default: https://qcloud-server.qpiai.tech)
"""

import os

# Try to load dotenv if available
try:
    from dotenv import load_dotenv

    # Load qcloud.env if it exists, otherwise fall back to .env
    if os.path.exists("qcloud.env"):
        load_dotenv("qcloud.env")
    else:
        load_dotenv()
except ImportError:
    # dotenv not available, but that's okay - we can still use os.getenv()
    pass

# Default server URL
DEFAULT_SERVER_URL = "https://qcloud-server.qpiai.tech"

# Bound both connection establishment and server response time for cloud requests.
HTTP_TIMEOUT = (10, 120)


def get_server_url() -> str:
    """
    Get the QPIAI server URL from environment variables.

    Returns:
        str: The server URL, defaults to https://qcloud-server.qpiai.tech

    Environment Variable:
        QPIAI_SERVER_URL: Base URL for the QPIAI Quantum server
    """
    server_url = os.getenv("QPIAI_SERVER_URL", DEFAULT_SERVER_URL)

    # Ensure URL doesn't end with a trailing slash for consistent usage
    return server_url.rstrip("/")


def get_sse_url(job_id: str) -> str:
    """
    Generate SSE URL for job status events.

    Args:
        job_id (str): The unique job identifier

    Returns:
        str: SSE endpoint URL for real-time job updates
    """
    server_url = get_server_url()
    return f"{server_url}/api/jobs/{job_id}/events"


def get_api_url(endpoint: str) -> str:
    """
    Generate full API URL for a given endpoint.

    Args:
        endpoint (str): API endpoint (e.g., '/api/circuits/create')

    Returns:
        str: Full URL for the API endpoint
    """
    server_url = get_server_url()

    # Ensure endpoint starts with a slash
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    return f"{server_url}{endpoint}"
