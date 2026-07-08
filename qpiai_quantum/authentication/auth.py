from typing import Optional
import os
import requests
from ..config import get_api_url
from .exceptions import (
    AuthError,
    APIKeyInvalidError,
    APIKeyEmptyError,
    UserNotFoundError,
)

from .user import set_user, get_user, clear_user, SDKUser


class QpiAIQuantumAuth:
    @staticmethod
    def _create_user_env():
        if os.environ.get("API_KEY", None) is None:
            raise APIKeyInvalidError()

    @staticmethod
    def login(api_key: Optional[str]) -> str:

        if api_key is None:
            raise APIKeyInvalidError()
        if not isinstance(api_key, str):
            raise APIKeyInvalidError()
        if api_key == "":
            raise APIKeyEmptyError()

        user_info = QpiAIQuantumAuth.me(api_key=api_key)
        user = SDKUser(
            name=user_info.get("name", ""),
            email=user_info.get("email", ""),
            api_key=api_key,
        )

        set_user(user)
        return "API key stored. It will be validated on first cloud access."

    @staticmethod
    def verify_api_key() -> bool:

        try:
            QpiAIQuantumAuth.me()
            return True
        except (APIKeyInvalidError, UserNotFoundError):
            return False
        except Exception:
            return False

    @staticmethod
    def logout() -> str:

        try:
            if get_user() is None:
                raise UserNotFoundError()
            clear_user()
            return "You have been logged out successfully."
        except Exception as e:
            raise AuthError(str(e))

    @staticmethod
    def me(api_key: Optional[str] = None) -> dict:

        try:
            if api_key is None:
                user = get_user()
                if user is None:
                    raise UserNotFoundError()
                current_api_key = user.api_key
            else:
                current_api_key = api_key

            headers = {"X-Secret-Token": current_api_key}
            url = get_api_url("/api/users/me")
            response = requests.get(url, headers=headers)

            if response.status_code == 401:
                raise APIKeyInvalidError("API key is invalid or expired")
            elif response.status_code == 403:
                raise APIKeyInvalidError("API key access forbidden")

            response.raise_for_status()
            user_data = response.json()

            return {
                "name": user_data.get("name", ""),
                "email": user_data.get("email", ""),
                "api_key": current_api_key,
            }

        except requests.exceptions.RequestException as e:
            raise AuthError(f"Failed to authenticate with server: {str(e)}")
        except Exception as e:
            raise AuthError(str(e))

    @staticmethod
    def list_compute_resources(display: bool = True):

        try:
            user = get_user()
            if user is None:
                raise UserNotFoundError()

            from ..jobmanager import JobManager

            job_manager = JobManager()
            resources = job_manager.list_compute_resources()

            filtered_resources = []
            for resource in resources:
                filtered_resource = {
                    "backend_name": resource.get("backend_name"),
                    "device_name": resource.get("device_name"),
                    "usage_rate": resource.get("usage_rate"),
                }
                filtered_resources.append(filtered_resource)

            if display:
                QpiAIQuantumAuth._display_resources(filtered_resources)
                return None
            else:
                return filtered_resources

        except UserNotFoundError:
            raise
        except Exception as e:
            raise AuthError(f"Failed to fetch compute resources: {str(e)}")

    @staticmethod
    def _display_resources(resources: list):

        if not resources:
            print("No compute resources available.")
            return

        print("\nAvailable Compute Resources:\n")

        for resource in resources:
            backend = resource.get("backend_name", "N/A")
            device_name = resource.get("device_name", "N/A")
            usage_rate = resource.get("usage_rate", "N/A")

            print(f"Backend: {backend}")
            print(f"  Device Name: {device_name}")
            print(f"  Usage Rate: {usage_rate} credits")
            print()
