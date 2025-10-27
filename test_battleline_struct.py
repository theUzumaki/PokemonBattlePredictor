"""
Example battleline struct for testing.
This creates a sample battleline structure matching the dataclasses in variables.py
"""

import variables as v

# Create example stats for Pokemon
starmie_stats = v.stats(
    atk=75,
    def_=85,
    spa=100,
    spd=100,
    spe=115,
    hp=60
)

exeggutor_stats = v.stats(
    atk=95,
    def_=85,
    spa=125,
    spd=125,
    spe=55,
    hp=95
)

chansey_stats = v.stats(
    atk=5,
    def_=5,
    spa=105,
    spd=105,
    spe=50,
    hp=250
)

# Create example moves
ice_beam = v.move(
    name="icebeam",
    cat=1,  # Special
    type="ice",
    base_pwr=95,
    accuracy=100,
    priority=0
)

psychic = v.move(
    name="psychic",
    cat=1,  # Special
    type="psychic",
    base_pwr=90,
    accuracy=100,
    priority=0
)

sleep_powder = v.move(
    name="sleeppowder",
    cat=0,  # Status
    type="grass",
    base_pwr=0,
    accuracy=75,
    priority=0
)

thunderbolt = v.move(
    name="thunderbolt",
    cat=1,  # Special
    type="electric",
    base_pwr=90,
    accuracy=100,
    priority=0
)

# Create example boosts (stat changes during battle)
neutral_boosts = v.stats(
    atk=0,
    def_=0,
    spa=0,
    spd=0,
    spe=0,
    hp=0
)

boosted_stats = v.stats(
    atk=1,
    def_=0,
    spa=2,
    spd=1,
    spe=0,
    hp=0
)

# Create example Pokemon
starmie = v.pkmn(
    id="starmie",
    hps=1.0,  # Full HP
    type1="water",
    type2="psychic",
    base_stats=starmie_stats,
    moves=[ice_beam, thunderbolt, psychic],
    status="nostatus",
    effects=["noeffect"],
    boosts=neutral_boosts
)

exeggutor = v.pkmn(
    id="exeggutor",
    hps=0.689,  # 68.9% HP
    type1="grass",
    type2="psychic",
    base_stats=exeggutor_stats,
    moves=[psychic, sleep_powder],
    status="frz",  # Frozen
    effects=["noeffect"],
    boosts=neutral_boosts
)

chansey = v.pkmn(
    id="chansey",
    hps=0.95,
    type1="normal",
    type2="notype",
    base_stats=chansey_stats,
    moves=[thunderbolt, ice_beam],
    status="nostatus",
    effects=["reflect"],
    boosts=boosted_stats
)

snorlax = v.pkmn(
    id="snorlax",
    hps=0.80,
    type1="normal",
    type2="notype",
    base_stats=v.stats(atk=110, def_=65, spa=65, spd=110, spe=30, hp=160),
    moves=[],
    status="nostatus",
    effects=["noeffect"],
    boosts=neutral_boosts
)

tauros = v.pkmn(
    id="tauros",
    hps=0.60,
    type1="normal",
    type2="notype",
    base_stats=v.stats(atk=100, def_=95, spa=70, spd=70, spe=110, hp=75),
    moves=[],
    status="par",  # Paralyzed
    effects=["noeffect"],
    boosts=neutral_boosts
)

alakazam = v.pkmn(
    id="alakazam",
    hps=0.40,
    type1="psychic",
    type2="notype",
    base_stats=v.stats(atk=50, def_=45, spa=135, spd=135, spe=120, hp=55),
    moves=[psychic],
    status="nostatus",
    effects=["noeffect"],
    boosts=neutral_boosts
)

# Create teams
team1 = v.team(
    pkmns=[starmie, chansey, snorlax, tauros, alakazam, 
           v.pkmn(id="filler", hps=1.0, type1="normal", type2="notype", 
                  base_stats=neutral_boosts, moves=[], status="nostatus", 
                  effects=[], boosts=neutral_boosts)]
)

team2 = v.team(
    pkmns=[exeggutor, chansey, starmie, alakazam, snorlax,
           v.pkmn(id="filler2", hps=0.5, type1="normal", type2="notype", 
                  base_stats=neutral_boosts, moves=[], status="nostatus", 
                  effects=[], boosts=neutral_boosts)]
)

team3 = v.team(
    pkmns=[tauros, snorlax, alakazam,
           v.pkmn(id="zapdos", hps=0.80, type1="electric", type2="flying",
                  base_stats=v.stats(atk=90, def_=85, spa=125, spd=90, spe=100, hp=90),
                  moves=[thunderbolt], status="nostatus", effects=["noeffect"], boosts=neutral_boosts),
           v.pkmn(id="rhydon", hps=0.30, type1="ground", type2="rock",
                  base_stats=v.stats(atk=130, def_=120, spa=45, spd=45, spe=40, hp=105),
                  moves=[], status="par", effects=["noeffect"], boosts=neutral_boosts),
           v.pkmn(id="lapras", hps=0.95, type1="water", type2="ice",
                  base_stats=v.stats(atk=85, def_=80, spa=125, spd=95, spe=60, hp=130),
                  moves=[ice_beam], status="nostatus", effects=["reflect"], boosts=boosted_stats)]
)

# Create adversary teams using the new adv_team structure
adv_team1 = v.adv_team(
    pkmn_alive=6,
    pkmn_dscvrd_alive=5,  # 5 out of 6 discovered
    types=["grass", "psychic", "ground", "rock", "electric", "flying", "ice", "fire"],
    statuses=["frz", "nostatus", "brn", "nostatus", "nostatus", "tox"],
    hp_leader=0.689  # HP of the leading Pokemon (exeggutor)
)

adv_team2 = v.adv_team(
    pkmn_alive=5,
    pkmn_dscvrd_alive=4,  # 4 out of 5 discovered
    types=["water", "psychic", "normal"],
    statuses=["nostatus", "nostatus", "par", "nostatus"],
    hp_leader=1.0  # HP of the leading Pokemon (starmie)
)

adv_team3 = v.adv_team(
    pkmn_alive=4,
    pkmn_dscvrd_alive=4,
    types=["fire", "flying", "dragon", "psychic"],
    statuses=["brn", "nostatus", "nostatus", "par"],
    hp_leader=0.45
)

adv_team4 = v.adv_team(
    pkmn_alive=6,
    pkmn_dscvrd_alive=3,
    types=["electric", "water", "ice", "normal"],
    statuses=["nostatus", "nostatus", "nostatus", "frz", "nostatus", "nostatus"],
    hp_leader=0.88
)

# Create battles with more variety
battle1 = v.battle(
    team1=team1,
    team2=adv_team1
)

battle2 = v.battle(
    team1=team1,
    team2=adv_team2
)

battle3 = v.battle(
    team1=team2,  # Different team1
    team2=adv_team3
)

battle4 = v.battle(
    team1=team3,  # Another different team1
    team2=adv_team4
)

# Create the battleline structure with more battles
example_battleline = v.battleline(
    battles={
        1: battle1,
        2: battle2,
        3: battle3,
        4: battle4
    }
)

def print_battleline_info(battleline: v.battleline):
    """Print information about the battleline structure."""
    print(f"Battleline contains {len(battleline.battles)} battles")
    print()
    
    for battle_id, battle in battleline.battles.items():
        print(f"Battle {battle_id}:")
        print(f"  Team 1: {len(battle.team1.pkmns)} Pokemon")
        for i, pkmn in enumerate(battle.team1.pkmns[:3], 1):  # Show first 3
            print(f"    {i}. {pkmn.id} ({pkmn.type1}/{pkmn.type2}) - HP: {pkmn.hps*100:.1f}%, Status: {pkmn.status}")
        
        print(f"  Adversary Team (Team 2):")
        print(f"    Pokemon Alive: {battle.team2.pkmn_alive}")
        print(f"    Pokemon Discovered Alive: {battle.team2.pkmn_dscvrd_alive}")
        print(f"    Types: {battle.team2.types}")
        print(f"    Statuses: {battle.team2.statuses}")
        print(f"    HP Leader: {battle.team2.hp_leader*100:.1f}%")
        print()

if __name__ == "__main__":
    print("Example Battleline Structure")
    print("=" * 50)
    print()
    
    print_battleline_info(example_battleline)
    
    # Access examples
    print("\nAccessing specific data:")
    print(f"Battle 1, Team 1, First Pokemon: {example_battleline.battles[1].team1.pkmns[0].id}")
    print(f"  HP: {example_battleline.battles[1].team1.pkmns[0].hps}")
    print(f"  Base Stats: ATK={example_battleline.battles[1].team1.pkmns[0].base_stats.atk}, "
          f"SPE={example_battleline.battles[1].team1.pkmns[0].base_stats.spe}")
    print(f"  Moves: {[move.name for move in example_battleline.battles[1].team1.pkmns[0].moves]}")
    print(f"  Boosts: SPA={example_battleline.battles[1].team1.pkmns[0].boosts.spa}")
    
    print("\nBattle 1, Adversary Team (Team 2):")
    print(f"  Pokemon Alive: {example_battleline.battles[1].team2.pkmn_alive}")
    print(f"  Pokemon Discovered Alive: {example_battleline.battles[1].team2.pkmn_dscvrd_alive}")
    print(f"  Types: {example_battleline.battles[1].team2.types}")
    print(f"  Statuses: {example_battleline.battles[1].team2.statuses}")
    print(f"  HP Leader: {example_battleline.battles[1].team2.hp_leader}")
