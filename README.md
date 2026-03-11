[Kanban Board](https://tree.taiga.io/project/brianleblanc98-powder-universe-simulation/kanban)

### Developer Setup (WIP!!)
- 

### Requirements (WIP!!)
- https://www.taichi-lang.org/

### Code Description (WIP!!)
1.
- SPH is used at fixed scales, this is an attempt to use SPH at variable scales in realtime, using Taichi Lang to run efficent simulations on GPUs. The idea is to simulate higher scales during longer time intervals (e.g. galaxy sized, 100s of millions of years), and lower scales during shorter time intervals (e.g. solar system, "realtime")

2.
- When flying the space ship, the game code will need the following:
    1. The **ship** itself
    2. [0-2] **central object(s)** (e.g. 0: interstellar space, 1: black hole/star, 2: binary stars)
    3. [0-8] **primary satellite(s)** (e.g. planets and others larger satellites)
        - These objects also have **secondy satellite(s)**, e.g. 1-30 small rocks
    4. ~[1-3] **notable satellite(s)** (e.g. comets, special secondary satellites, rouge planets when in interstellar space)
    5. ~[100-1000] **minor satellite(s)** (e.g. asteroids, kuiper belt objects)
    6. All the gas/dust
    7. All the radiation (mostly electro-magnatic, i.e. light/radio-waves/gamma-rays/x-rays, but gravitational as well [gravity also radiates through waves... it's complicated])
        - Note: For point 6 and 7, they'll probably be best modeled as large fields rather than individual particles. Dust alone would be on the scale of hundreds of thousands of particles. I'm not entirely sure what the best solution is, but points 1-5 should be enough to work on for now
- For each of the **ship**, **central object(s)**, and **satellites(s)**, we'll need to store the **position**, **velocity**, and **acceleration**. These values will be calculated in other code (i.e. call a function "getPosition()" from a seperate engine code which calls my physics code), and then communicated back to the game code.
- The **central object(s)** dominate the environment, think of a black hole ripping apart a neutron star, but also all the gas/dust and radiation play a major role in game play. Proto-stars are surrounded by the material that eventually forms planets, surely a hostile environment to navigate. Nebulae could host interesting mechanics by the virtue of extreme radiation making ship components malfunction. Supernovae are the most energetic explosions *IN THE ENTIRE UNIVESRE* (That is not a joke, the smartest minds cannot conceive of any way to create more energy in a single moment).
- The **notable satellites** are the "Points of Interest" for this game. The best loot would be located here.

3. **"Wand"(Space Ship) System**
- Basically I'm ripping off Noita, but applying the the logic to space ship parts (**"parts"**, henceforth) instead of spells. I also don't want to go as in depth as Noita does. A simpler system of the same kind should be enough to get the tinkering feeling that Noita gives while not being overwhelming.
- Straight from [The Noita Wiki](https://noita.wiki.gg/wiki/Spells), the attributes of a spell are:
    1. Uses (If it has limited uses)
    2. Mana Drain
    3. Cast Delay
    4. Recharge Time
    5. Spread Modification
    6. Radius
    7. Speed
    8. Crit Chance Bonus
    9. Damage
- These all translate pretty much perfectly to **parts**, so instead we have:
    1. **Uses** (If it has limited uses)
    2. **Resource Drain**
    3. **Firing Delay**
    4. **Recharge Time**
    5. **Spread Modification**
    6. **Radius**
    7. **Speed**
    8. **Crit Chance Bonus**
    9. **Damage**
    - (Descriptions of these are not complete)
- Instead of constantly finding wands (ships) and swapping spells (parts), 1-3 ships per "run" is enough. The player upgrades the statistics of the ship over time, and finds materials to craft the parts (I have no idea what "finds materials" and "craft" mean yet).
- Again, straight from [The Noita Wiki](https://noita.wiki.gg/wiki/Wands), the attributes of a wand are:
    1. Shuffle
    2. Spells/Cast
    3. Cast Delay
    4. Recharge Time
    5. Mana Max
    6. Mana Charge Speed
    7. Capacty
    8. Spread
    9. Always Casts
    10. Speed Multiplier
- In my opinion, shuffle is not a fun mechanic for most people. Making a good *no* shuffle wand takes careful consideration, and I think that it can be too much for a lot of people. There's a lot going on in this game already (e.g. Fighting enemies, maneouvering in combat, *and orbital mechanics*), so I think the ship modification system should have a high skill floor. Not including shuffle accomplishes that.
- Moving on, the **statistics of a ship** are:
    1. TBD

### Citations

- Price, Daniel J. "Smoothed particle hydrodynamics and magnetohydrodynamics." Journal of Computational Physics 231.3 (2012): 759-794. (https://arxiv.org/abs/1012.1885)

- Price, Daniel J., et al. "Phantom: A smoothed particle hydrodynamics and magnetohydrodynamics code for astrophysics." Publications of the Astronomical Society of Australia 35 (2018): e031. (https://arxiv.org/abs/1702.03930)

- Liptai, David, and Daniel J. Price. "General relativistic smoothed particle hydrodynamics." Monthly Notices of the Royal Astronomical Society 485.1 (2019): 819-842. (https://arxiv.org/abs/1901.08064)

- Laibe, Guillaume, and Yona Lapeyre. "The Shamrock code: I-Smoothed Particle Hydrodynamics on GPUs." arXiv preprint arXiv:2503.09713 (2025). (https://arxiv.org/abs/2503.09713)

- Cullen, Lee, and Walter Dehnen. "Inviscid smoothed particle hydrodynamics." Monthly Notices of the Royal Astronomical Society 408.2 (2010): 669-683. (https://arxiv.org/abs/1006.1524)

- Duffell, Paul C., et al. "The Santa Barbara Binary− disk Code Comparison." The Astrophysical Journal 970.2 (2024): 156. (https://arxiv.org/abs/2402.13039)

- Schaal, Kevin, et al. "Astrophysical hydrodynamics with a high-order discontinuous Galerkin scheme and adaptive mesh refinement." Monthly Notices of the Royal Astronomical Society 453.4 (2015): 4278-4300. (https://arxiv.org/abs/1506.06140)

- Müller, Bernhard. "Hydrodynamics of core-collapse supernovae and their progenitors." Living Reviews in Computational Astrophysics 6.1 (2020): 3.