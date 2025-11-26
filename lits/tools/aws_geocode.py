from pydantic import Field
from .base import BaseTool
from ..clients.als_client import AmazonLocationClient

class AWSGeocodeTool(BaseTool):
    name:str = "AWS_Geocode"
    description:str = "Use this tool to convert an address/place name into geographic coordinates (latitude and longitude) using AWS Location Service. Input should be a string containing the address/place name to geocode."

    def __init__(self, client: AmazonLocationClient=None):
        client = client or AmazonLocationClient()
        super().__init__(client=client)

    def _run(self, address: str):
        geocode_result = self.client.request(address)
        return geocode_result
    
    def _arun(self, address: str):
        raise NotImplementedError("This tool does not support async")
