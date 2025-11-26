import sys
sys.path.append('.')
import unittest

# Test coverage:
# - Valid JSON payload without wrappers.
# - JSON embedded in a quoted string.
# - Auto-repair of truncated JSON missing closing braces.
# - Failure raised when payload is irreparably malformed.

from langagent.agents.utils import parse_json_string


class ParseToolActionTests(unittest.TestCase):
    def test_parses_well_formed_json(self):
        payload = '{"action": "Search", "action_input": {"query": "coffee"}}'
        parsed = parse_json_string(payload)
        self.assertEqual(parsed["action"], "Search")
        self.assertEqual(parsed["action_input"]["query"], "coffee")

    def test_handles_wrapped_string_payload(self):
        payload = '"{\\"action\\": \\"Lookup\\", \\"action_input\\": {\\"id\\": 42}}"'
        parsed = parse_json_string(payload)
        self.assertEqual(parsed["action"], "Lookup")
        self.assertEqual(parsed["action_input"]["id"], 42)

    def test_repairs_missing_closing_brace(self):
        payload = '{"action": "NearbyPlaces", "action_input": {"radius": 1000'
        parsed = parse_json_string(payload)
        self.assertEqual(parsed["action"], "NearbyPlaces")
        self.assertEqual(parsed["action_input"]["radius"], 1000)

    def test_raises_on_irreparable_json(self):
        payload = '{"action": "Broken", "action_input": {"oops": }'
        with self.assertRaises(ValueError):
            parse_json_string(payload)


if __name__ == "__main__":
    unittest.main()
