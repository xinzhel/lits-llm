import requests
import logging
from dataclasses import asdict, dataclass
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

def inspect_toolkit(tools):
    for tool in tools:
        inspect_tool(tool)
        
def inspect_tool(tool):
    print("Type: ", type(tool))
    print("Name:", tool.name) # multiply
    print("Description:", tool.description) # Multiply two numbers.
    print("Args:", tool.args)
    print("\n")
    
def test_connection():
    # find_nearby = NearbyPlacesTool(client=client)
    # find_nearby.invoke({"placeId": "0", "type": "accounting" })
    # run the following command to find the Bearer token:
    # curl -X POST http://10.224.245.233:5000/api/login -H "Content-Type: application/json" -d '{"username": "test", "password": "123"}'
    url = "http://10.224.245.233:5000/api"+"/map/nearby"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJpc3MiOiJtYXBxdWVzdC1hcHAub25yZW5kZXIuY29tIiwiaWF0IjoxNzU5OTk4NzUyfQ.ss23UkIzD73vogcFoRoUG1GrBAfFTOB0_H2CAS6Q-Z0"
    }
            
    params = {
        "location": 0,
        "radius": None,
        "type": "accounting",
        "keyword": None,
        "rankby": "distance"
    }
    response = requests.get(url, headers=headers, params=params)
    print(response)