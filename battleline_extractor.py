import pandas as pd
import json
def create_final_turn_feature(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    counter = 0
    battles = [] # populate, for now singlura team
    for battle in data:
        features = {}
        p1, p2 = battle['p1_team_details'], battle['p2_lead_details']
        teams = InitTeams(p1, p2)
       
        # explore turns 
        for turn in battle['battle_timeline']:
            hp1, hp2 = ExploreTurns(turn)
            teams[0][turn['p1_pokemon_state']['name']] = hp1
            teams[1][turn['p2_pokemon_state']['name']] = hp2

        
        teams[0]['discovered'] = 6
        teams[1]['discovered'] = len(teams[1])

        lead_hp_1 = battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct']
        lead_hp_2 = battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct']

        teams[0]['lead_hp'] = lead_hp_1
        teams[1]['lead_hp'] = lead_hp_2

        
        return teams

    
# returns the hp percentage of p1 and p2 at current turn
def ExploreTurns(turn):
    p1 = turn['p1_pokemon_state']['hp_pct']
    p2 = turn['p2_pokemon_state']['hp_pct']
    return p1, p2
    

def InitTeams(p1_details, p2_details):
    teams = {0: {}, 1: {}}

    # p1_details is already a list
    for poke in p1_details:
        key = poke['name']
        teams[0][key] = 1.0

    # ensure p2_details is a list
    if isinstance(p2_details, dict):
        p2_details = [p2_details]
    
    for poke in p2_details:
        key = poke['name']
        teams[1][key] = 1.0
    

    return teams


if __name__ == "__main__":
    train_data = []


    with open("data/train.json", 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            train_data.append(json.loads(line))

    print(f"Successfully loaded {len(train_data)} battles.")

    
   
    print(create_final_turn_feature(train_data))