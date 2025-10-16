from typing import Union, Iterator, Dict, Any
from pathlib import Path
import json

def iter_test_data(path: Union[str, Path] = None, entry_to_print = 0, SECOND_TEAM = False) -> tuple[Iterator[Dict[str, Any]], Dict[str, int]]:
    """
    Yield JSON objects from a JSONL file one by one (memory efficient).
    Additionally, count occurrences of Pokémon in 'p1_team_details'.
    """
    if path is None:
        path = Path(__file__).resolve().parent / ".." / "data" / "train.jsonl"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    pokemon_count = {}

    entries = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)

                if "p1_team_details" in entry and isinstance(entry["p1_team_details"], list):
                    for pokemon in entry["p1_team_details"]:
                        if isinstance(pokemon, dict) and "name" in pokemon:
                            pokemon_name = pokemon["name"]
                            if isinstance(pokemon_name, str):
                                pokemon_count[pokemon_name] = pokemon_count.get(pokemon_name, 0) + 1

                if SECOND_TEAM:
                    pokemon = entry["p2_lead_details"]
                    if isinstance(pokemon, dict) and "name" in pokemon:
                        pokemon_name = pokemon["name"]
                        if isinstance(pokemon_name, str):
                            pokemon_count[pokemon_name] = pokemon_count.get(pokemon_name, 0) + 1

                if lineno <= entry_to_print:
                    entry_str = json.dumps(entry, indent=2)
                    if len(entry_str) > 500:  # Truncate if exceeds 500 characters
                        entry_str = entry_str[:500] + "...\n[Truncated]"
                    print(f"Entry {lineno}:\n{entry_str}")

            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {e.msg}") from e

    return entries, pokemon_count
