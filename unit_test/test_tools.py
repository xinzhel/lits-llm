import sys
sys.path.append('../')
from lits.clients.als_client import AmazonLocationClient
from dotenv import load_dotenv

def test_gocode():
    load_dotenv()
    REGION = "ap-southeast-2"
    client = AmazonLocationClient(
        region=REGION,
    )
    coords = client.request("Melbourne CBD")
    print(f"📍 Coordinates (longitude, latitude): {coords} (type: {type(coords)})")
    print(client.ping())
    # client.delete_place_index('MyPlaceIndex')
    
