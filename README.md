# Projet MiniLab

Le dépôt d’exercice du cours [MiniLab LLM](https://github.com/stephanemalho/minilab-llm) :
vous y construisez, sur votre propre processeur, un vrai petit modèle de langage,
du tokenizer jusqu’au serveur qui répond en streaming.

Le cours en ligne vous guide pas à pas, partie « Projet MiniLab ». Ce dépôt contient
seulement ce dont vous avez besoin pour démarrer :

```
packages/
  minilab/
    data/sft_fr.jsonl              les dialogues du fine-tuning (étape 5)
    tokenizer.py … server.py       à écrire vous-même, une étape à la fois
  minillm/data/
    verne_tour_du_monde.txt        le corpus : Le Tour du monde en quatre-vingts jours
tests/minilab/                     les tests qui valident chaque étape
```

## Installer

Il faut Git, Python 3.12 ou plus récent et [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/stephanemalho/minilab-projet.git
cd minilab-projet
uv sync
uv run python -c "import torch; print(torch.__version__)"
```

## Construire une étape

1. Écrivez le fichier de l’étape dans `packages/minilab/` (par exemple `tokenizer.py`).
2. Lancez-le pour voir les vrais calculs : `uv run python -m packages.minilab.tokenizer`.
3. Validez-le : `uv run pytest tests/minilab/test_tokenizer.py`.

Les étapes, dans l’ordre : `tokenizer`, `model`, `train`, `generate`, `chat`, `server`.

## Du rouge au vert

Au départ, les 30 tests sont rouges : aucun fichier d’étape n’existe, donc pytest ne peut
même pas les charger (`6 errors during collection`). C’est normal : les tests sont écrits
avant le code, et votre travail consiste à les faire passer au vert, étape par étape.
Ne modifiez jamais les tests : si l’un d’eux échoue, c’est le code qu’il faut corriger.

Pour suivre votre progression sur tout le projet :

```sh
uv run pytest tests/minilab -q --continue-on-collection-errors
```

La dernière ligne passe de `6 errors` à `6 passed, 5 errors` après l’étape 1, puis jusqu’à
`30 passed` quand le projet est complet.

Les fichiers produits (tokenizer, poids du modèle) sont écrits dans `checkpoints/minilab/`,
ignoré par Git.

## Le serveur

```sh
uv run uvicorn packages.minilab.server:app --port 8001
```

## Licence du corpus

Le texte de Jules Verne (1873) est dans le domaine public ; voir
`packages/minillm/data/README.md`.
