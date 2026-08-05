# Navigation visuelle par apprentissage par renforcement dans une scène 3D Gaussian Splatting

Portage PyTorch du modèle acteur-critique siamois de Zhu et al. (ICRA 2017), appliqué
non plus au simulateur AI2-THOR mais à la scène `room` du dataset Mip-NeRF 360,
reconstruite par 3D Gaussian Splatting.

L'agent apprend à rejoindre une cible visuelle donnée sous forme d'image, en se
déplaçant dans un graphe de poses discrètes construit à partir de la scène reconstruite.

## Principe

La scène 3DGS est rendue depuis une grille discrète de positions et d'orientations.
Chaque pose devient un nœud du graphe de navigation, décrit par ses features ResNet-50.
L'agent n'observe jamais la scène brute : il manipule uniquement ces vecteurs de 2048
dimensions, ce qui rend l'entraînement rapide sans avoir à faire tourner un moteur de
rendu.

Le réseau est *target-driven* : l'état courant et l'image cible passent par les mêmes
couches partagées (structure siamoise), puis par des couches spécifiques à la scène.
Changer de cible ne demande donc pas de réentraîner le modèle entier.

- **Actions** : `MoveForward`, `MoveBackward`, `RotateLeft`, `RotateRight`
- **Récompenses** : `+10` à la cible, `-0.1` en cas de collision, `-0.01` par pas
- **Algorithme** : A3C, un thread par cible, gradients appliqués sur un réseau global
  partagé via RMSProp

## Format du graphe de navigation

Le fichier `scene_3dgs.h5` suit le format des dumps d'AI2-THOR, avec un champ
supplémentaire :

| Dataset | Forme | Contenu |
| --- | --- | --- |
| `resnet_feature` | (N, 1, 2048) | features ResNet-50 de chaque pose |
| `graph` | (N, 4) | transitions : `graph[i][j]` = nœud atteint par l'action `j` depuis `i`, `-1` si collision |
| `shortest_path_distance` | (N, N) | distances en nombre de pas entre paires de nœuds |
| `location` | (N, 2) | coordonnées (x, y) sur la grille |
| `poses_names` | (N,) | noms des poses, encodés en bytes |

Le fichier n'est pas versionné dans ce dépôt (228 Mo). Il est produit par le notebook
de construction du graphe à partir des rendus 3DGS et des poses COLMAP.

## Structure

```
.
├── train.py             boucle d'entraînement A3C multi-thread
├── evaluate.py          évaluation : 100 épisodes par cible
├── network.py           réseau acteur-critique siamois
├── scene_loader.py      environnement discret lisant le HDF5
├── training_thread.py   thread A3C : collecte, calcul de la loss, gradients
├── accum_trainer.py     accumulation des gradients locaux
├── rmsprop_applier.py   RMSProp partagé entre threads
├── visualize.py         animation GIF d'un épisode (vue + carte 2D)
├── keyboard_agent.py    navigation manuelle dans la scène
├── constants.py         hyperparamètres et liste des cibles
└── utils/
```

## Configuration

Tout se règle dans `constants.py`. Les paramètres qui comptent :

```python
TASK_LIST = {
    'scene_3dgs': ['308', '126', '274', '147', '306']   # indices des cibles
}
PARALLEL_SIZE  = 20          # doit être >= au nombre de cibles
MAX_TIME_STEP  = 10.0 * 10**6
HISTORY_LENGTH = 4           # nombre de frames empilées
CHECKPOINT_DIR = 'checkpoints'
```

Chaque thread entraîne une cible ; avec 20 threads et 5 cibles, 4 threads travaillent
sur chaque cible. Les indices de `TASK_LIST` doivent être des nœuds valides du graphe,
donc inférieurs à `N`.

`HISTORY_LENGTH = 4` détermine la taille d'entrée du réseau : 2048 × 4 = 8192, valeur
codée dans la première couche de `network.py`. Modifier l'un impose de modifier l'autre,
et rend les checkpoints existants inutilisables.

`CHECKPOINT_DIR` doit pointer sur un dossier propre à cette scène. Un checkpoint issu
d'un entraînement AI2-THOR n'est pas chargeable ici : le nombre de scènes et la taille
d'entrée diffèrent, et `load_state_dict()` échoue au démarrage.

## Utilisation

```bash
pip install -r requirements.txt

python train.py       # entraînement
python evaluate.py    # évaluation sur les cibles de TASK_LIST
```

Le chemin du HDF5 est défini en tête de `train.py`, `evaluate.py` et
`scene_loader.py` — à adapter avant tout lancement.

Les checkpoints sont écrits tous les 10 000 pas dans `CHECKPOINT_DIR`, et rechargés
automatiquement au démarrage : un entraînement interrompu reprend où il s'est arrêté.

## Exécution sur cluster SLURM

```bash
#!/bin/bash
#SBATCH --job-name=a3c_navigation
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32


PYTHON=$HOME/.conda/envs/<env>/bin/python
export LD_LIBRARY_PATH=$HOME/.conda/envs/<env>/lib:$LD_LIBRARY_PATH

cd <chemin>/icra2017-visual-navigation-1
$PYTHON -u train.py
```

Point appris à l'usage :

**`Illegal instruction (core dumped)`** — les binaires PyTorch compilés par EasyBuild
sont incompatibles avec le jeu d'instructions CPU des nœuds GPU. La solution est
d'appeler le Python de l'environnement conda par son chemin absolu, avec un
`LD_LIBRARY_PATH` explicite, et **sans** `module load`.
