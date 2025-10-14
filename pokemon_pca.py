from pathlib import Path
import json
from typing import List, Dict, Any, Iterator, Union
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict

pokemon_species = []
pokemon_species_count = defaultdict(int)  # Counter for Pokémon species occurrences
feature_names = ["name", "atk", "def"]

def iter_test_data(path: Union[str, Path] = None, entry_to_print = 0) -> Iterator[Dict[str, Any]]:
    """
    Yield JSON objects from a JSONL file one by one (memory efficient).
    """
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "train.jsonl"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                if lineno <= entry_to_print:
                    entry = json.loads(line)
                    entry_str = json.dumps(entry, indent=2)
                    if len(entry_str) > 500:  # Truncate if exceeds 500 characters
                        entry_str = entry_str[:500] + "...\n[Truncated]"
                    print(f"Entry {lineno}:\n{entry_str}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {e.msg}") from e
            
def name_to_int(name: str) -> int:
    if name in pokemon_species:
        pokemon_species_count[name] += 1  # Increment the counter for this Pokémon
        return pokemon_species.index(name)
    else:
        pokemon_species.append(name)
        pokemon_species_count[name] += 1  # Initialize the counter for this Pokémon
        return len(pokemon_species) - 1

# Extract numerical data for PCA
def extract_features(data: List[Dict[str, Any]], max_elements: int = float("inf")) -> np.ndarray:
    features = {}
    for feat in feature_names:
        features[feat] = []
        
    i = 0
    for record in data:
        if i >= max_elements:
            break
        i += 1

        for pokemon in record["p1_team_details"]:
            for key in feature_names:
                format_key = f"base_{key}" if key != "name" else "name"
                if format_key not in pokemon:
                    raise KeyError(f"Missing key '{key}' in Pokémon data: {pokemon}")

                if key == "name":
                    features["name"].append(name_to_int(pokemon[format_key]))
                else:
                    features[key].append(pokemon[f"{format_key}"])
    return np.column_stack([features[key] for key in features])

if __name__ == "__main__":
    # quick check when running the module directly
    data = list(iter_test_data())
    components = 2

    print("\n\n---- Data Summary ----\n")
    print(f"Loaded {len(data)} records from data/test.jsonl")


    # Perform PCA
    features = extract_features(data, max_elements=10000)

    print(len(pokemon_species), "unique Pokemon species found.\n")
    for i, species in enumerate(pokemon_species):
        print(f"{i + 1:3}: {species}\t(Count: {pokemon_species_count[species]})")
        if i >= 20:
            print("...")
            break
    print()

    if features.size > 0:
        pca = PCA(n_components=components)
        transformed = pca.fit_transform(features)

        # Retrieve the names of the best features
        best_features = [feature_names[i] for i in pca.components_[0].argsort()[-components:][::-1]]
        print("Best features and contributes to first principal component:")
        for i in range(len(best_features)):
            percentage = pca.explained_variance_ratio_[i] * 100
            print(f"{best_features[i]}\t-\t{percentage:.2f}%")
    else:
        print("No valid features found for PCA.")

    print("\n---- End of Summary ----\n\n")