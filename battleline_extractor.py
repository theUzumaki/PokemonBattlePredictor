import pandas as pd
import json


# CLASSES
from variables import pkmn, move
from variables import team, stats, adv_team
from variables import battle as Battle
from variables import battleline as Battleline

def create_final_turn_feature(data: list[dict], is_train: bool = True) -> pd.DataFrame:
    """Create a battleline struct from raw JSON battle records.

    Args:
        data: list of battle dicts (parsed from JSONL)
        is_train: if True, expects each record to contain 'player_won' and will
                  populate the battle.win label. If False, the label will be set
                  to -1 to indicate unknown (useful for test data).

    Returns:
        Battleline dataclass instance containing parsed battles.
    """

    battleline = Battleline(battles = {})
    for i, battle in enumerate(data):
        # If training data, read the label; otherwise set to -1 (unknown)
        if is_train:
            win_label = battle['player_won']
        else:
            win_label = -1

        battle_ = Battle(team1=None, team2=None, win=win_label)
        #print(battle)
        party1_details, party2_details = battle['p1_team_details'], battle['p2_lead_details']
        team1 = init_team_1(party1_details)
        team2 = init_adv_team()
        #battle_.team1 = team1
        #battle_.team2 = team2
        
        found_in_starting_team= set()
        found_in_adv_team = set()


        # moves 
        team1_moves = set()
        team2_moves = set()

        #status 
        team1_statusses = set()
        team2_statusses = set()
        # explore turns 
        for turn in battle['battle_timeline']:
            poke1 = turn['p1_pokemon_state']
            poke2 = turn['p2_pokemon_state']
            
            hp1, hp2 = return_turn_pokemon_hp_perc(poke1, poke2)
            idx_team = returnPokemonFromName(team1.pkmns, poke1['name'])
            team1.pkmns[idx_team].hps = hp1 


            # POKEMON ALIVE
            if poke2['name'] not in found_in_adv_team:
                team2.pkmn_dscvrd_alive += 1
            if hp2 == 0.0:
                team2.pkmn_dscvrd_alive -= 1
                team2.pkmn_alive -= 1
            if poke1['name'] not in found_in_starting_team:
                found_in_starting_team.add(poke1['name'])

            # POKEMON MOVES
            move_ = parseTurnMovesTeam1(turn)
            if move_ != None:
                if move_.name not in team1_moves:
                    team1_moves.add(move_.name)
                    team1.pkmns[idx_team].moves.append(move(
                        name=move_.name,
                        cat= 1 if move_.cat == 'physical' else 0,
                        type=move_.type,
                        base_pwr=move_.base_pwr,
                        accuracy=move_.accuracy,
                        priority= move_.priority
                    ))
            move2_ = parseTurnMovesTeam2(turn)
            if move2_ != None:
                if move2_.name not in team2_moves:
                    team2_moves.add(move2_.name)
                    team2.types.append(move(
                        name=move2_.name,
                        cat= 1 if move2_.cat == 'physical' else 0,
                        type=move2_.type,
                        base_pwr=move2_.base_pwr,
                        accuracy=move2_.accuracy,
                        priority= move2_.priority
                    ))

          

        lead_hp_2 = battle['battle_timeline'][-1]['p2_pokemon_state']['hp_pct']
        team2.hp_leader = lead_hp_2
       
        team1.revealed = len(found_in_starting_team)
        battle_.team1 = team1
        battle_.team2 = team2
        battleline.battles[i] = battle_
    return battleline   
def parse_turn_status_team1(poke1_state):
    if poke1_state['status'] == 'nostatus':
        return None
    return poke1_state['status']
def parse_turn_status_team2(poke2_state):
    pass
def parseTurnMovesTeam2(turn):
    if turn['p2_move_details'] == None: # skips no move 
        return None
   
    move_ = turn['p2_move_details']
    return move(
        name=move_['name'],
        cat=move_['category'],
        type=move_['type'],
        base_pwr=move_['base_power'],
        accuracy=move_['accuracy'],
        priority= move_['priority']
    ) 
def parseTurnMovesTeam1(turn):
    if turn['p1_move_details'] == None: # skips no move 
        return None
   
    move_ = turn['p1_move_details']
    return move(
        name=move_['name'],
        cat=move_['category'],
        type=move_['type'],
        base_pwr=move_['base_power'],
        accuracy=move_['accuracy'],
        priority= move_['priority']
    )
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

def init_adv_team():
    team2 =   adv_team(
        pkmn_alive=6,
        pkmn_dscvrd_alive=1,
        types=[],
        statuses=[],
        hp_leader=1.0
    )
    return team2
    
     
def init_team_1(p1_details):
    team1 = team(pkmns=[], revealed=0)
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
            boosts= stats(
                atk=0, 
                def_=0, 
                spa=0, 
                spd=0, 
                spe=0, 
                hp=0
            )
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
