# OPTIMAX // Super Simulateur OR

Simulateur de jeu pedagogique pour ateliers de Recherche Operationnelle:

- digital twin multi-quartiers
- moteur Monte Carlo (30-300 scenarios par tour)
- choix politiques (mandats + priorites)
- arbitrage multi-objectifs (performance, robustesse, clarte, social)
- historique exportable CSV

## Lancer avec Docker

```bash
docker compose up --build
```

Puis ouvrir `http://localhost:8501`.

## Arreter

```bash
docker compose down
```

## Structure

- `app.py`: simulateur principal (moteur + UI)
- `data/city.json`: topologie ville, quartiers, etat initial
- `data/modules.json`: 10 modules OR
- `data/events.json`: cartes evenement
- `data/roles.json`: roles et priorites
- `Dockerfile`: image runtime
- `docker-compose.yml`: orchestration locale
