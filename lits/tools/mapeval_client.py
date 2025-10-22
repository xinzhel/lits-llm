import requests
from typing import Dict, Any, Optional
import time
from ..tools.config import ServiceConfig
from ..tools.constant import mapeval_types

class MapEvalClient:
    """MapEval API 客户端封装"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': config.bearer_token
        })
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """统一请求处理"""
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault('timeout', self.config.timeout)
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        
        # 模拟原代码的延迟
        # time.sleep(30)
        print(f"Response from {url}: {response.text}")
        return response.json()
    
    def search_place(self, query: str) -> Dict[str, Any]:
        """搜索地点"""
        print("Searching place with query:", query)
        text = self._request('GET', '/map/search', params={'query': query})
        return text
    
    def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """获取地点详情"""
        return self._request('GET', f'/map/details/{place_id}')
    
    def get_nearby_places(self, location: str, place_type: str, 
                         rankby: str = 'distance', 
                         radius: Optional[int] = None) -> Dict[str, Any]:
        """获取附近地点"""
        params = {
            'location': location,
            "radius": None if rankby == 'distance' else radius,
            'type': place_type if place_type in mapeval_types else None,
            "keyword": place_type if place_type not in mapeval_types else None,
            'rankby': rankby,
        }
        return self._request('GET', '/map/nearby', params=params)
    
    def get_travel_time(self, origin: str, destination: str, mode: str) -> Dict[str, Any]:
        """获取旅行时间"""
        params = {
            'origin': origin,
            'destination': destination,
            'mode': mode
        }
        return self._request('GET', '/map/distance/custom', params=params)
    
    def get_directions(self, origin: str, destination: str, mode: str) -> Dict[str, Any]:
        """获取路线"""
        params = {
            'origin': origin,
            'destination': destination,
            'mode': mode
        }
        return self._request('GET', '/map/directions', params=params)