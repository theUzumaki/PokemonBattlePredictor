import json
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Search for a specific variable in the JSON data.")
parser.add_argument("variable", type=str, help="The variable to search for (e.g., 'effects').")
args = parser.parse_args()

# Load the JSON data from data/train.jsonl
parsed_data = []
with open('data/train.jsonl', 'r') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            parsed_data.append(json.loads(line))

# Extract unique values for the specified variable
unique_values = set()

for data_entry in parsed_data:
    for event in data_entry.get("battle_timeline", []):
        for player, player_data in event.items():  # Iterate over players in the event
            if isinstance(player_data, dict) and args.variable in player_data:
                result = player_data[args.variable]
                unique_values.add(result)

# Print the unique values
print(unique_values)