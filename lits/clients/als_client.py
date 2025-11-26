import os
import requests
import logging
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from .base import BaseClient


logger = logging.getLogger(__name__)

class AmazonLocationClient(BaseClient):
    def __init__(self, api_key=None, iam_credentials=None, region="ap-southeast-2"):
        """
        Initialize the Amazon Location Service client.
        Supports either:
          - API key (for browser / app integrations)
          - IAM credentials (for internal AWS environment)
        Automatically creates a Place Index if missing in IAM mode.
        """
        self.region = region
        # --- Configure session ---
        if iam_credentials is None:
            iam_credentials = {
                "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID") or None,
                "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY") or None,
                "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN") or None,
            }

        if iam_credentials['AWS_ACCESS_KEY_ID'] and iam_credentials['AWS_SECRET_ACCESS_KEY'] and iam_credentials.get('AWS_SESSION_TOKEN'):
            boto3.setup_default_session(
                aws_access_key_id=iam_credentials['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=iam_credentials['AWS_SECRET_ACCESS_KEY'],
                aws_session_token=iam_credentials['AWS_SESSION_TOKEN'],
                region_name=region,
            )
            self.index_name = "MaptalkPlaceIndex"
            self.iam_credentials = iam_credentials
        else:
            self.iam_credentials = None

        self.api_key = api_key or os.getenv("AMAZON_LOCATION_API_KEY") or None
        if self.api_key:
            assert self.iam_credentials is None, "Cannot use both API key and IAM credentials."
            self.index_name = "PlaceIndexDefault-Esri"
        logger.info(f"ℹ️ API Key mode: {self.api_key is not None}, IAM mode: {self.iam_credentials is not None}")
        # --- Initialize client ---
        self.client = boto3.client(
            "location",
            config=Config(
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
        
    def delete_place_index(self, index_name):
        """Delete a Place Index by name."""
        logger.info("ℹ️ Existing Place Indexes: %s", self.list_place_indexes())
        self.client.delete_place_index(IndexName=index_name)
        logger.info("Existing Place Indexes after deletion: %s", self.list_place_indexes())
        
    def list_place_indexes(self):
        """List existing Place Indexes."""
        response = self.client.list_place_indexes()
        index_names = [index["IndexName"] for index in response.get("Entries", [])]
        return index_names
    
    def request(self, place_name):
        """ Geocode using Amazon Location Service.
        """
        if self.iam_credentials:
            return self._get_coords_with_iam(place_name)
        else:
            return self._get_coords_with_requests(place_name)
        
    def _get_coords_with_iam(self, place_name):
        """ Supports either:
        - API key (for browser / app integrations)
        - IAM credentials (for internal AWS environment)
        Automatically creates a Place Index if missing in IAM mode.
        """

        # --- Prepare request parameters ---
        params = {
            "IndexName": self.index_name,
            "Text": place_name,
            "MaxResults": 1,
        }
        if self.api_key:
            params["Key"] = self.api_key

        # --- Perform geocoding ---
        try:
            response = self.client.search_place_index_for_text(**params)
        except ClientError as e:
            logger.error("❌ Error calling SearchPlaceIndexForText: %s", e)
            return None

        # --- Extract coordinates safely ---
        results = response.get("Results", [])
        if not results:
            logger.warning(f"⚠️ No results found for place name: {place_name}.")
            return None

        coords = results[0]["Place"]["Geometry"]["Point"]
        return coords


    def _get_coords_with_requests(self, place_name):
        """ 
        Retrieves the longitude and latitude coordinates for a given place name 
        using the AWS Location Service Geocode API.

        Args:
            place_name (str): The name of the location to search for (e.g., "Melbourne").
            
        Returns:
            list[float, float] or None: A list containing the coordinates 
                                        [Longitude, Latitude] of the best match, 
                                        or None if no results are found.
        
        Example raw response.json:
            {
                'ResultItems': [{
                    'PlaceId': 'AQAAADoABzCXJdab5mgo-SZRWmpGkUpkRvvbZEYkIzVqdoXxN0WhyC6RGOd8gjz7lWCijrPiP0BNHkrFLdVf8IE1i0WOV8A7EtODX_TPqQ4acnT6pLy0CmgGKUDG8M9G', 
                    'PlaceType': 'Locality', 
                    'Title': 'Melbourne, VIC 3004, Australia', 
                    'Address': {
                        'Label': 'Melbourne, VIC 3004, Australia', 
                        'Country': {'Code2': 'AU', 'Code3': 'AUS', 'Name': 'Australia'}, 
                        'Region': {'Code': 'VIC', 'Name': 'Victoria'}, 
                        'Locality': 'Melbourne', 
                        'PostalCode': '3004'
                    }, 
                    'Position': [144.96755, -37.81736], 
                    'MapView': [144.5531, -38.22504, 145.54978, -37.50209], 
                    'MatchScores': {
                        'Overall': 0.75, 
                        'Components': {'Address': {'Locality': 1}}
                    }, 
                    'ParsedQuery': {
                        'Address': {'Locality': [{'StartIndex': 0, 'EndIndex': 9, 'Value': 'Melbourne', 'QueryComponent': 'Query'}]}
                    }
                }]
            }

        Example final return value:
            [144.96755, -37.81736]
        """

        # https://docs.aws.amazon.com/general/latest/gr/location.html
        # url = f"https://places.geo.{REGION}.amazonaws.com/geocode?key={API_KEY}"
        # url = f"https://places.geo.{region}.api.aws/geocode?key={api_key}"
        url = f"https://places.geo.{self.region}.amazonaws.com/v2/geocode?key={self.api_key}"
        payload = {
            "QueryText": place_name,
            "MaxResults": 1
        }
        resp = requests.post(url, json=payload)
        
        # This will raise requests.exceptions.HTTPError for 4xx or 5xx status codes, i.e., `resp.status_code` is not 200
        resp.raise_for_status()
        
        resp_data = resp.json()
        
        # Check if results are present and extract the Position array
        if resp_data.get('ResultItems'):
            # The coordinates are in the 'Position' key of the first item in 'ResultItems'
            return resp_data['ResultItems'][0]['Position']
        else:
            # Handle cases where no results are found (e.g., return None, or an empty list)
            return None # Or []
        
    def ping(self):
        """ Simple test to verify connectivity to AWS Location Service. """
        try:
            response = self.client.list_place_indexes()
            logger.info("✅ Successfully connected to AWS Location Service. Place Indexes: %s", 
                        [index["IndexName"] for index in response.get("Entries", [])])
            return True
        except ClientError as e:
            logger.error("❌ Failed to connect to AWS Location Service: %s", e)
            return False