# Navigation visuelle target-driven par apprentissage par renforcement (AI2-THOR)


## Principe

Chaque scène est fournie sous forme de dump HDF5 : une grille discrète de positions,
échantillonnées dans les quatre directions cardinales, et pour chacune la feature
ResNet-50 de la vue first-person correspondante. L'agent ne manipule que ces vecteurs de
2048 dimensions, jamais le simulateur — d'où un entraînement rapide.

Le réseau est *target-driven* : l'état courant et l'image cible traversent les mêmes
couches partagées (structure siamoise), puis des couches spécifiques à la scène.
Changer de cible ne demande pas de réentraîner le modèle entier.

- **Actions** : `MoveForward`, `RotateRight`, `RotateLeft`, `MoveBackward`
- **Récompenses** : `+10` à la cible, `-0.1` en cas de collision, `-0.01` par pas
- **Algorithme** : A3C, un thread par couple (scène, cible), gradients appliqués sur un
  réseau global partagé via RMSProp

## Structure

```
.
├── train.py                    boucle d'entraînement A3C multi-thread
├── evaluate.py                 100 épisodes par cible, longueur moyenne des trajectoires
├── network.py                  réseau acteur-critique siamois
├── scene_loader.py             environnement discret lisant les dumps HDF5
├── training_thread.py          thread A3C : collecte, loss, gradients
├── visualize.py                animation GIF d'un épisode (vue + carte 2D)
├── keyboard_agent.py           navigation manuelle dans une scène
├── constants.py                hyperparamètres, scènes et cibles
├── utils/
│   ├── accum_trainer.py        accumulation des gradients locaux
│   └── rmsprop_applier.py      RMSProp partagé entre threads
└── data/                       dumps de scène (non versionnés)
```

## Données

Les dumps de scène ne sont pas inclus dans ce dépôt (77 à 289 Mo chacun). Ils se
téléchargent depuis le site du premier auteur :

```bash
wget http://vision.stanford.edu/yukezhu/thor_v1_scene_dumps.zip -P data/
cd data && unzip thor_v1_scene_dumps.zip && rm thor_v1_scene_dumps.zip
```

On obtient les quatre scènes attendues :

```
data/bathroom_02.h5  bedroom_04.h5  kitchen_02.h5  living_room_08.h5
```

Chaque dump contient, ligne par ligne :

| Dataset | Contenu |
| --- | --- |
| `observation` | vue first-person 300×400×3 |
| `resnet_feature` | feature ResNet-50 de 2048 dimensions |
| `location` | coordonnées (x, y) sur une grille au pas de 0,5 m |
| `rotation` | orientation de l'agent : 0, 90, 180 ou 270° |
| `graph` | transitions : `graph[i][j]` = état atteint par l'action `j` depuis `i`, `-1` si collision |
| `shortest_path_distance` | distances en nombre de pas, `-1` si inaccessible |

## Configuration

Tout se règle dans `constants.py` :

```python
TASK_LIST = {
    'bathroom_02'    : ['26', '37', '43', '53', '69'],
    'bedroom_04'     : ['134', '264', '320', '384', '387'],
    'kitchen_02'     : ['90', '136', '157', '207', '329'],
    'living_room_08' : ['92', '135', '193', '228', '254']
}
PARALLEL_SIZE  = 100         # doit être >= au nombre total de couples (scène, cible)
MAX_TIME_STEP  = 100.0 * 10**6
HISTORY_LENGTH = 4           # nombre de features empilées
CHECKPOINT_DIR = 'checkpoints'
```

`train.py` construit dynamiquement la liste des branches à partir de `TASK_LIST` et
vérifie que `PARALLEL_SIZE` est suffisant. Avec 20 couples et 100 threads, 5 threads
travaillent sur chaque couple.

`HISTORY_LENGTH = 4` détermine la taille d'entrée du réseau : 2048 × 4 = 8192, valeur
codée dans la première couche de `network.py`.

## Utilisation

```bash
pip install -r requirements.txt

python train.py       # entraînement
python evaluate.py    # évaluation sur les cibles de TASK_LIST
```



Un GIF est produit tous les 500 épisodes dans `animations/`, et les récompenses sont
journalisées dans `rewards_log.csv`.

## Exécution sur cluster SLURM

```bash
#!/bin/bash
#SBATCH --job-name=a3c_navigation
#SBATCH --output=$HOME/logs/output_%j.log
#SBATCH --error=$HOME/logs/error_%j.log
#SBATCH --partition=gpu
#SBATCH --account=<votre-compte>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32

mkdir -p $HOME/logs

PYTHON=<chemin absolu du python de l'environnement conda>
export LD_LIBRARY_PATH=<chemin des lib de cet environnement>:$LD_LIBRARY_PATH

echo "--- Test de validation PyTorch et CUDA ---"
$PYTHON -c "import torch; print('Version PyTorch:', torch.__version__); print('GPU CUDA disponible ?:', torch.cuda.is_available()); print('Nom du GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Aucun')"

echo "--- Lancement de l'entraînement ---"
$PYTHON -u train.py

```

Point appris à l'usage :

**`Illegal instruction (core dumped)`** — les binaires PyTorch compilés par EasyBuild
sont incompatibles avec le jeu d'instructions CPU des nœuds GPU. La solution est
d'appeler le Python de l'environnement conda par son chemin absolu, avec un
`LD_LIBRARY_PATH` explicite, et **sans** `module load`.
ctions, signalées par des commentaires `BUG FIX`
dans les sources :

