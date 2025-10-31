import pandas as pd
import json


# CLASSES
from variables import pkmn
from variables import team, stats, adv_team

def create_final_turn_feature(data: list[dict]) -> pd.DataFrame:
    feature_list = []
    counter = 0
    battles = [] # populate, for now singlura team
    found_in_starting_team = set()
    for battle in data:
        features = {}
        #print(battle)
        party1_details, party2_details = battle['p1_team_details'], battle['p2_lead_details']
        team1 = init_team_1(party1_details)
        team2 = init_adv_team(party2_details)
        '''
        # explore turns 
        for turn in battle['battle_timeline']:
            poke1 = turn['p1_pokemon_state']
            poke2 = turn['p2_pokemon_state']
            
            hp1, hp2 = return_turn_pokemon_hp_perc(poke1, poke2)
            team1[poke1['name']] = hp1
            #teams[1][poke2['name']] = hp2
            if poke1['name'] not in found_in_starting_team:
                found_in_starting_team.add(poke1['name'])

        # assign discovered pokemons for both teams 
        #teams[0]['discovered'] = len(found_in_starting_team)
        #teams[1]['discovered'] = len(teams[1])

        # last seen pokemon battle layout
        lead_hp_1 = battle['battle_timeline'][-1]['p1_pokemon_state']['hp_pct']
        lead_hp_2 = battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']

        #teams[0]['lead_hp'] = lead_hp_1
        #teams[1]['lead_hp'] = lead_hp_2

        '''
        return pd.DataFrame(team1.pkmns)
    
def returnPokemonFromName(team, pokemon_name):
    for i, pkm in enumerate(team):
        if pkm.id == pokemon_name:
            return i 
    
    return None
    
# returns the hp percentage of p1 and p2 at current turn
def return_turn_pokemon_hp_perc(pokemon_state1, pokemon_state2):
    hp1 = pokemon_state1['hp_pct']
    hp2 = pokemon_state2['hp_pct']
    return hp1, hp2

def init_adv_team(p2_details):
    team2 =   adv_team(
        pkmn_alive=6,
        pkmn_dscvrd_alive=1,
        types=[],
        statuses=[],
        hp_leader=1.0
    )
    
     
def init_team_1(p1_details):
    team1 = team(pkmns=[])
    poke_list = []
    # p1_details is already a list
    for poke in p1_details:
        key = poke['name']
        poke = pkmn(
            id=key, 
            hps=1.0,
            type1=poke['types'][0],
            type2=poke['types'][1],
            base_stats=stats(
                atk=poke['base_atk'],
                def_= poke['base_def'],
                spa= poke['base_spa'],
                spd=poke['base_spd'],
                spe=poke['base_spe'],
                hp=poke['base_hp']
            ),
            moves=[],
            status= [],
            effects=[],
            boosts=[]
        )
        poke_list.append(poke)
        

    team1.pkmns = poke_list
    return team1
   


if __name__ == "__main__":
    train_data = []


    with open("data/train.json", 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            train_data.append(json.loads(line))

    print(f"Successfully loaded {len(train_data)} battles.")

    
   
    print(create_final_turn_feature(train_data))
