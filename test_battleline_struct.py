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
    pkmns=[exeggutor, 
           v.pkmn(id="rhydon", hps=0.85, type1="ground", type2="rock",
                  base_stats=v.stats(atk=130, def_=120, spa=45, spd=45, spe=40, hp=105),
                  moves=[], status="nostatus", effects=["noeffect"], boosts=neutral_boosts),
           v.pkmn(id="golem", hps=0.70, type1="rock", type2="ground",
                  base_stats=v.stats(atk=120, def_=130, spa=55, spd=65, spe=45, hp=80),
                  moves=[], status="brn", effects=["noeffect"], boosts=neutral_boosts),
           v.pkmn(id="zapdos", hps=1.0, type1="electric", type2="flying",
                  base_stats=v.stats(atk=90, def_=85, spa=125, spd=90, spe=100, hp=90),
                  moves=[], status="nostatus", effects=["noeffect"], boosts=neutral_boosts),
           v.pkmn(id="articuno", hps=0.55, type1="ice", type2="flying",
                  base_stats=v.stats(atk=85, def_=100, spa=125, spd=125, spe=85, hp=90),
                  moves=[], status="nostatus", effects=["noeffect"], boosts=neutral_boosts),
           v.pkmn(id="moltres", hps=0.45, type1="fire", type2="flying",
                  base_stats=v.stats(atk=100, def_=90, spa=125, spd=85, spe=90, hp=90),
                  moves=[], status="tox", effects=["noeffect"], boosts=neutral_boosts)]
)

# Create a battle
battle1 = v.battle(
    team1=team1,
    team2=team2
)

battle2 = v.battle(
    team1=team2,  # Swap teams for variety
    team2=team1
)

# Create the battleline structure
example_battleline = v.battleline(
    battles={
        1: battle1,
        2: battle2
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
        
        print(f"  Team 2: {len(battle.team2.pkmns)} Pokemon")
        for i, pkmn in enumerate(battle.team2.pkmns[:3], 1):  # Show first 3
            print(f"    {i}. {pkmn.id} ({pkmn.type1}/{pkmn.type2}) - HP: {pkmn.hps*100:.1f}%, Status: {pkmn.status}")
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
    
    print("\nBattle 1, Team 2, First Pokemon: {example_battleline.battles[1].team2.pkmns[0].id}")
    print(f"  Status: {example_battleline.battles[1].team2.pkmns[0].status}")
    print(f"  Effects: {example_battleline.battles[1].team2.pkmns[0].effects}")
