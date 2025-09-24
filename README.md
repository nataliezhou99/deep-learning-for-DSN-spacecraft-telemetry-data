
# 2D Platformer Game – *The Floor Is Lava*

This repository contains a **browser-playable 2D platformer** written in C using SDL2.  
It was built as a portfolio project to learn low-level game development, physics and collision detection,  
and WebAssembly via Emscripten.  

The goal of the game is simple yet challenging: **guide your character across a series of platforms,  
avoid hazards like lava and water, collect gems, and reach the door to advance levels.**

---

## Gameplay
- Side-scrolling 2D platformer
- Physics engine with gravity, jumping, and collisions
- Hazards: lava pits, water, moving blocks
- Collectible gems to increase score
- Multiple interactive levels with buttons, doors, and elevators
- Victory condition: reach the exit door

---

## Features
- **Custom Engine**  
  - Physics system (bodies, forces, collision detection via Separating Axis Theorem)  
  - Scene graph for managing entities and assets  
  - Vector math utilities
- **Asset Management**  
  - Cached loading of sprites, animations, sounds, and music  
  - Modular asset system for reuse across levels
- **Graphics & Audio**  
  - Built with SDL2 for rendering and audio  
  - Background music and sound effects
- **Cross-Platform**  
  - Runs locally (Linux/Mac)  
  - Exportable to the browser using Emscripten (WebAssembly)

---

## Code Structure
```

include/     # Header files (engine, physics, scene, assets)
library/     # Engine implementation (physics, collision, SDL wrapper, asset cache)
demo/        # Game logic (level generation, event loop, scoring, win conditions)
assets/      # Sprites, textures, music, and demo screenshot
Makefile     # Build rules (local + web builds)

````

---

## Build & Run

### Prerequisites
- **SDL2** development libraries
- **clang** or **gcc**
- (Optional) **Emscripten** for web builds

### Local Build
```bash
make
./bin/game
````

### Web Build

```bash
make game
python3 -m http.server
```

Then open `http://localhost:8000/demo/game.html` in your browser.

---

## Extending the Game

This project is designed to be **modular**:

* Add new assets (sprites, music) to the `assets/` folder
* Create additional levels by editing `demo/game.c`
* Define new physics objects and forces via `include/body.h` and `include/forces.h`

---

## Skills Demonstrated

* Low-level memory management in C
* Game physics (forces, collisions, SAT algorithm)
* SDL2 graphics and audio
* Modular design and caching systems
* Cross-compilation to WebAssembly
* Git version control & Makefile automation

---

## Authors

Developed by **Natalie Zhou, Olivia Wang, and Dillan Lau**


Natalie Zhou
* Computer Science @ Caltech
* Interests: game development, systems programming, applied machine learning
* Contact: [LinkedIn](https://www.linkedin.com/in/nataliezhou99) | [GitHub](https://github.com/nataliezhou99)
