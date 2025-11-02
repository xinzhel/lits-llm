import requests
from typing import Dict, Any
from .base import BaseClient

class RestApiClient(BaseClient):
    """Generic REST API client."""

    def __init__(self, base_url: str, headers: Dict[str, str] = None, timeout: int = 30, bearer_token: str = None):
        super().__init__(base_url=base_url, timeout=timeout)
        self.session = requests.Session()
        self.session.headers.update(headers or {})
        if bearer_token:
            if bearer_token.startswith('Bearer '):
                bearer_token = bearer_token[7:]
            self.session.headers.update({
                'Authorization': f'Bearer {bearer_token}'
            })
        self.base_url = base_url
        self.timeout = timeout

    def request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.timeout)
        resp = self.session.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> bool:
        try:
            resp = self.session.get(self.base_url, timeout=3)
            return resp.ok
        except Exception:
            return False
