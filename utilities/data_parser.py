from typing import Union, Iterator, Dict, Any, List
from pathlib import Path
import json

def iter_test_data(path: Union[str, Path] = None, entry_to_print = 0, SECOND_TEAM = False, fields_to_look_into: List[str] = None) -> tuple[Iterator[Dict[str, Any]], Dict[str, int]]:
    """
    Yield JSON objects from a JSONL file one by one (memory efficient).
    Additionally, count occurrences of Pokémon by traversing the specified field path.
    
    Args:
        path: Path to the JSONL file
        entry_to_print: Number of entries to print for debugging
        SECOND_TEAM: Whether to include second team's Pokémon
        fields_to_look_into: List of nested field names to traverse (e.g., ["p1_team_details"])
    """
    if path is None:
        path = Path(__file__).resolve().parent / ".." / "data" / "train.jsonl"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    if fields_to_look_into is None:
        fields_to_look_into = ["p1_team_details"]

    entry_count = {}

    entries = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)

                # Navigate through the nested fields
                current_data = entry
                field_found = True
                
                for field in fields_to_look_into:
                    if isinstance(current_data, dict) and field in current_data:
                        current_data = current_data[field]
                    else:
                        field_found = False
                        break
                
                # Process the data at the target field
                if field_found and isinstance(current_data, list):
                    for entry in current_data:
                        if isinstance(entry, dict):
                            key = ''.join(str(value) for value in entry.values())
                            if key in entry_count:
                                entry_count[key] += 1
                            else:
                                entry_count[key] = 1

                if lineno <= entry_to_print:
                    entry_str = json.dumps(entry, indent=2)
                    if len(entry_str) > 500:  # Truncate if exceeds 500 characters
                        entry_str = entry_str[:500] + "...\n[Truncated]"
                    print(f"Entry {lineno}:\n{entry_str}")

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {e.msg}") from e

    return entries, entry_count
