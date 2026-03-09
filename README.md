# MCV C6 Project (Team 1)

## Members

- Diego Hernández Antón
- Oriol Juan Sabater
- Valentin Micu Hontan
- Xavier Pacheco Bach
- Benet Ramió Comas

## Quick Setup

This project uses Git submodules. Clone the repository as below:
```bash
git clone --recurse-submodules https://github.com/mcv-m6-video/mcv-c6-2026-team1.git
cd mcv-c6-2026-team1/
```

If you cloned the repository without submodules, run:
```bash
git submodule update --init --recursive
```

Setup the environment:

```bash
./setup.sh
conda activate c6-team1
```

## Project Structure

`WeekX/` contains everything developed during week `X`.

```bash
mcv-c6-2026-team1/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.sh
├── WeekX/
│   ├── data/      # Data for Week X
│   ├── src/       # Code for Week X
└── └── README.md  # Documentation for Week X
```
