from pydantic import BaseModel, Field
from .base import BaseTool
from typing import Type, Optional
import inflect
from ..clients.mapeval_client import MapEvalClient

p = inflect.engine()

# ===== Input Schemas =====
class PlaceSearchInput(BaseModel):
    placeName: str = Field(description="Name and address of the place")

class PlaceDetailsInput(BaseModel):
    placeId: str = Field(description="Place Id of the location")

class NearbyPlacesInput(BaseModel):
    placeId: str = Field(description="The id of the place around which to retrieve nearby places.")
    type: str = Field(description="Type of place (e.g., restaurant, hospital, etc). Restricts the results to places matching the specified type.")
    rankby: str = Field(default='distance', description="Specifies the order in which places are listed. Possible values are: (1. prominence (default): This option sorts results based on their importance. When prominence is specified, the radius parameter is required. 2. distance: This option sorts places in ascending order by their distance from the specified location. When distance is specified, radius is disallowed. In case you are not concerned about the radius, use rankby as distance.)")
    radius: Optional[int] = Field(default=None, description="Defines the distance (in meters) within which to return place results.")

class TravelTimeInput(BaseModel):
    originId: str = Field(description="Place Id of Origin")
    destinationId: str = Field(description="Place Id of Destination")
    travelMode: str = Field(description="Mode of transportation (driving, walking, bicycling, transit)")

class DirectionsInput(BaseModel):
    originId: str = Field(description="Place Id of Origin")
    destinationId: str = Field(description="Place Id of Destination")
    travelMode: str = Field(description="Mode of transportation (driving, walking, bicycling, transit)")


# ===== Tool Implementations =====
class PlaceSearchTool(BaseTool):
    name: str = "PlaceSearch"
    description: str = "Get place ID for a given location name and address."
    args_schema: Type[BaseModel] = PlaceSearchInput
    handle_tool_error: bool = True
    
    client: MapEvalClient = Field(description="API client instance")
    
    def __init__(self, client: MapEvalClient, **kwargs):
        super().__init__(client=client, **kwargs)
    
    def _run(self, placeName: str) -> str:
        try:
            data = self.client.search_place(placeName)
            if data.get('results') and len(data['results']) > 0:
                return data['results'][0]['place_id']
            else:
                return "Incorrect place name. Please use the same name as in the question."
        except Exception as e:
            return f"Error searching place: {str(e)}"

class PlaceDetailsTool(BaseTool):
    name: str = "PlaceDetails"
    description: str = "Get detailed information for a given place ID."
    args_schema: Type[BaseModel] = PlaceDetailsInput
    handle_tool_error: bool = True
    
    client: MapEvalClient = Field(description="API client instance")
    
    def __init__(self, client: MapEvalClient, **kwargs):
        super().__init__(client=client, **kwargs)
    
    def _run(self, placeId: str) -> str:
        try:
            place = self.client.get_place_details(placeId)['result']
            return self.place_to_context(place)
        except Exception as e:
            return f"Incorrect Place ID or error: {str(e)}"
    
    def place_to_context(self, place: dict) -> str:
        text = ""
        
        # 位置信息
        lat = place['geometry']['location'].get('lat', 'N/A')
        lng = place['geometry']['location'].get('lng', 'N/A')
        text += f"- Location: {place.get('formatted_address', '')} ({lat}, {lng})\n"
        
        # 联系方式
        if place.get('phone_number'):
            text += f"- Phone Number: {place['phone_number']}\n"
        
        # 营业时间
        if place.get('opening_hours', {}).get('weekday_text'):
            text += f"- Open: {', '.join(place['opening_hours']['weekday_text'])}\n"
        
        # 评分
        if place.get('rating'):
            text += f"- Rating: {place.get('rating', '')} ({place.get('user_ratings_total', 0)} ratings)\n"
        
        # 价格等级
        if place.get('price_level'):
            price_map = ["Free", "Inexpensive", "Moderate", "Expensive", "Very Expensive"]
            price_level = price_map[place.get('price_level', 0)]
            text += f"- Price Level: {price_level}\n"
            
        # 其他信息
        if place.get("delivery"):
            text += "- Delivery Available.\n" if place.get('delivery') else "- Delivery Not Available.\n"

        if place.get("dine_in"):
            text += "- Dine In Available.\n" if place.get('dine_in') else "- Dine In Not Available.\n"

        if place.get("reservable"):
            text += "- Reservable.\n" if place.get('reservable') else "- Not Reservable.\n"

        if place.get("serves_breakfast"):
            text += "- Serves Breakfast.\n" if place.get('serves_breakfast') else "- Does Not Serve Breakfast.\n"

        if place.get("serves_lunch"):
            text += "- Serves Lunch.\n" if place.get('serves_lunch') else "- Does Not Serve Lunch.\n"

        if place.get("serves_dinner"):
            text += "- Serves Dinner.\n" if place.get('serves_dinner') else "- Does Not Serve Dinner.\n"

        if place.get("takeout"):
            text += "- Takeout Available.\n" if place.get('takeout') else "- Takeout Not Available.\n"

        if place.get("wheelchair_accessible_entrance"):
            text += "- Wheelchair Accessible Entrance.\n" if place.get('wheelchair_accessible_entrance') else "- Not Wheelchair Accessible Entrance.\n"

        
        return text

class NearbyPlacesTool(BaseTool):
    name: str = "NearbyPlaces"
    description: str = "Get nearby places around a location."
    args_schema: Type[BaseModel] = NearbyPlacesInput
    handle_tool_error: bool = True
    
    client: MapEvalClient = Field(description="API client instance")
    
    def __init__(self, client: MapEvalClient, **kwargs):
        super().__init__(client=client, **kwargs)
    
    def _run(self, placeId: str, type: str, rankby: str = 'distance', 
             radius: Optional[int] = None) -> str:
        if rankby == "distance" and radius is not None and radius > 0:
            return "When rankby is distance, radius is disallowed. If want to use rankby as distance, please set radius to 0. And if you want to use radius, please set rankby as prominence."
        
        try:
            data = self.client.get_nearby_places(placeId, type, rankby, radius)
            if data.get('results'):
                return self._format_nearby_places(data['results'], type, rankby, radius)
            else:
                return "No nearby places found."
        except Exception as e:
            return f"Error finding nearby places: {str(e)}"
    
    def _format_nearby_places(self, places: list, place_type: str, 
                             rankby: str, radius: Optional[int]) -> str:
        """格式化附近地点"""
        text = f"Nearby {p.plural(place_type)} "
        text += f"({'in ' + str(radius) + 'm radius' if rankby == 'prominence' else 'sorted by distance'}):\n"
        
        for i, place in enumerate(places, 1):
            text += f"{i}. {place.get('name', 'Unknown')} ({place.get('place_id')})\n"
            if place.get('vicinity'):
                text += f"   - Address: {place['vicinity']}\n"
            if place.get('rating'):
                text += f"   - Rating: {place['rating']} ({place.get('user_ratings_total', 0)} ratings)\n"
        
        return text

class TravelTimeTool(BaseTool):
    name: str = "TravelTime"
    description: str = "Estimate the travel time between two places."
    args_schema: Type[BaseModel] = TravelTimeInput
    handle_tool_error: bool = True
    
    client: MapEvalClient = Field(description="API client instance")
    
    def __init__(self, client: MapEvalClient, **kwargs):
        super().__init__(client=client, **kwargs)
    
    def _run(self, originId: str, destinationId: str, travelMode: str) -> str:
        try:
            data = self.client.get_travel_time(originId, destinationId, travelMode)
            
            if data['matrix'][0][0].get('duration') is None:
                return "No route found. Please check the place ids and try again."
            
            duration = data['matrix'][0][0]['duration']['text']
            distance = data['matrix'][0][0]['distance']['text']
            
            mode_map = {
                'transit': 'public transport',
                'driving': 'car',
                'bicycling': 'cycle',
                'walking': 'foot'
            }
            mode_text = mode_map.get(travelMode.lower(), travelMode)
            
            return f"Travel time by {mode_text} is {duration} ({distance})."
        except Exception as e:
            return f"Error: {str(e)}"
        
    def _arun(self, originId: str, destinationId: str, travelMode: str):
        raise NotImplementedError("This tool does not support async")


class DirectionsTool(BaseTool):
    name: str = "Directions"
    description: str = "Get detailed directions/routes between two places."
    args_schema: Type[BaseModel] = DirectionsInput
    handle_tool_error: bool = True
    
    client: MapEvalClient = Field(description="API client instance")
    
    def __init__(self, client: MapEvalClient, **kwargs):
        super().__init__(client=client, **kwargs)
    
    def _run(self, originId: str, destinationId: str, travelMode: str) -> str:
        try:
            data = self.client.get_directions(originId, destinationId, travelMode)
            
            if len(data['routes']) == 0:
                return "No route found. Please check the place ids and try again."
            
            return self._format_directions(data['routes'], travelMode)
        except Exception as e:
            return f"Error getting directions: {str(e)}"
    
    def _format_directions(self, routes: list, mode: str) -> str:
        """格式化路线信息"""
        mode_map = {
            'transit': 'public transport',
            'driving': 'car',
            'bicycling': 'cycle',
            'walking': 'foot'
        }
        mode_text = mode_map.get(mode.lower(), mode)
        
        text = f"There are {len(routes)} routes by {mode_text}. They are:\n"
        
        for i, route in enumerate(routes):
            leg = route['legs'][0]
            text += f"{i+1}. Via {route['summary']} | {leg['duration']['text']} | {leg['distance']['text']}\n"
            for step in leg['steps']:
                text += f"   - {step['html_instructions']}\n"
        
        return text