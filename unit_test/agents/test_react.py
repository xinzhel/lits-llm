import sys, os
root_dir = os.getcwd()
sys.path.append(root_dir)
from lits.tools.mapeval_tools import PlaceSearchTool, NearbyPlacesTool
from lits.agents.utils import execute_tool_action
from dotenv import dotenv_values, load_dotenv
load_dotenv()
from lits.tools import build_tools


def test_mapeval_action():
    from lits.clients.mapeval_client import MapEvalClient
    uri = "http://localhost:5000/api"
    bearer_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6InRlc3QiLCJpc3MiOiJtYXBxdWVzdC1hcHAub25yZW5kZXIuY29tIiwiaWF0IjoxNzYwMTg4Mjg5fQ.i7zPvYEIkGoQAcx4D6SW171jRhKfYbioXe5hgmfWyfQ"
    
    # Initialize MapEvalClient
    client = MapEvalClient(base_url=uri, bearer_token=bearer_token)

    action_str = '{\n "action": "PlaceSearch",\n "action_input": {\n "placeName": " Cusco, Peru"\n }\n}'
    tools = [PlaceSearchTool(client), NearbyPlacesTool(client)]
    obs = execute_tool_action(action_str, tools)
    print("Tool action result:", obs)
    return obs
# addr, addrfeat, bg, county, county_lookup, countysub_lookup, cousub, direction_lookup, edges, faces, featnames, geocode_settings, geocode_settings_default, layer, loader_lookuptables, loader_platform, loader_variables, pagc_gaz, pagc_lex, pagc_rules, place, place_lookup, secondary_unit_lookup, spatial_ref_sys, state, state_lookup, street_type_lookup, tabblock, tabblock20, topology, tract, zcta5, zip_lookup, zip_lookup_all, zip_lookup_base, zip_state, zip_state_loc",
# addr, addrfeat, bg, county, county_lookup, countysub_lookup, cousub, direction_lookup, edges, faces, featnames, geocode_settings, geocode_settings_default, layer, loader_lookuptables, loader_platform, loader_variables, pagc_gaz, pagc_lex, pagc_rules, place, place_lookup, secondary_unit_lookup, spatial_ref_sys, state, state_lookup, street_type_lookup, tabblock, tabblock20, topology, tract, zcta5, zip_lookup, zip_lookup_all, zip_lookup_base, zip_state, zip_state_loc
def test_veris_action():
    action_str = "{\n\"action\": \"sql_db_list_tables\",\n\"action_input\": {}\n}"
    tools = build_tools(benchmark_name="geosql")
    obs = execute_tool_action(action_str, tools)
    print("Tool action result:", obs)
    return obs

if __name__ == "__main__":
    obs = test_veris_action()
    print("Final observation:", obs)
    