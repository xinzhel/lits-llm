from typing import Dict, Any, Optional
from dataclasses import dataclass
from .web_client import RestApiClient

mapeval_types = [
	"accounting",
	"airport",
	"amusement_park",
	"art_gallery",
	"atm",
	"bakery",
	"bank",
	"bar",
	"beauty_salon",
	"bicycle_store",
	"book_store",
	"bowling_alley",
	"bus_station",
	"cafe",
	"campground",
	"car_dealer",
	"car_rental",
	"car_repair",
	"car_wash",
	"casino",
	"cemetery",
	"church",
	"city_hall",
	"clothing_store",
	"convenience_store",
	"courthouse",
	"dentist",
	"department_store",
	"doctor",
	"drugstore",
	"electrician",
	"electronics_store",
	"embassy",
	"fire_station",
	"florist",
	"food",
	"funeral_home",
	"furniture_store",
	"gas_station",
	"gym",
	"hair_care",
	"hardware_store",
	"health",
	"hindu_temple",
	"home_goods_store",
	"hospital",
	"insurance_agency",
	"jewelry_store",
	"laundry",
	"lawyer",
	"library",
	"light_rail_station",
	"liquor_store",
	"local_government_office",
	"locksmith",
	"lodging",
	"meal_delivery",
	"meal_takeaway",
	"mosque",
	"movie_rental",
	"movie_theater",
	"moving_company",
	"museum",
	"night_club",
	"painter",
	"park",
	"parking",
	"pet_store",
	"pharmacy",
	"physiotherapist",
	"place_of_worship",
	"plumber",
	"point_of_interest",
	"police",
	"political",
	"post_office",
	"primary_school",
	"real_estate_agency",
	"restaurant",
	"roofing_contractor",
	"rv_park",
	"school",
	"secondary_school",
	"shoe_store",
	"shopping_mall",
	"spa",
	"stadium",
	"storage",
	"store",
	"subway_station",
	"supermarket",
	"taxi_stand",
	"tourist_attraction",
	"train_station",
	"transit_station",
	"travel_agency",
	"university",
	"veterinary_care",
	"zoo"
]

class MapEvalClient(RestApiClient):
    """MapEval API 客户端封装"""
    
    def search_place(self, query: str) -> Dict[str, Any]:
        """搜索地点"""
        print("Searching place with query:", query)
        text = self.request('GET', '/map/search', params={'query': query})
        return text
    
    def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """获取地点详情"""
        return self.request('GET', f'/map/details/{place_id}')
    
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
        return self.request('GET', '/map/nearby', params=params)
    
    def get_travel_time(self, origin: str, destination: str, mode: str) -> Dict[str, Any]:
        """获取旅行时间"""
        params = {
            'origin': origin,
            'destination': destination,
            'mode': mode
        }
        return self.request('GET', '/map/distance/custom', params=params)
    
    def get_directions(self, origin: str, destination: str, mode: str) -> Dict[str, Any]:
        """获取路线"""
        params = {
            'origin': origin,
            'destination': destination,
            'mode': mode
        }
        return self.request('GET', '/map/directions', params=params)