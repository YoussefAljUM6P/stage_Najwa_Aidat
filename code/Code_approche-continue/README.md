# Asservissement visuel par apprentissage par renforcement (A3C / PPO)

Ce dossier contient la version script de l'approche continue.
Il regroupe une série d'expériences d'asservissement visuel dans une scène 3D
reconstruite par **Gaussian Splatting** (scène *room* de Mip-NeRF 360) : un agent
contrôle une caméra virtuelle (translation + rotation) et doit rejoindre une pose
cible. Deux algorithmes acteur-critique sont comparés, **A3C** et **PPO**.

Les expériences vont d'une formulation « pose → pose » (l'agent connaît sa pose et
celle de la cible) jusqu'à une formulation **visuelle**, où l'état est la différence
entre les features ResNet-18 de l'image courante et de l'image cible, toutes deux
rendues par gsplat.

## Structure

```
RL-visual-servoing/
├── README.md
├── requirements.txt
├── donnees/
│   ├── splat.ply                    modèle Gaussian Splatting de la scène « room »
│   └── dataparser_transforms.json   transformation nerfstudio (repère COLMAP -> repère du splat)
├── config.py        chemins, intrinsèques caméra, lecture de la transformation nerfstudio
├── gs_scene.py      chargement du .ply, conversions de poses, rendu gsplat
├── envs.py          les 7 environnements Gymnasium
├── models.py        ActorCriticMLP, ActorCritic, ResNetFeatureExtractor
├── algorithms.py    boucles d'entraînement A3C et PPO
├── plots.py         figures de suivi
├── train.py         point d'entrée : une expérience = une section du notebook
├── evaluate.py      évaluation des politiques (fonctions + CLI sur checkpoint)
└── visualize.py     grille des images du jeu de données, rendu d'une pose
```

## Données

Le dossier `donnees/` contient les deux fichiers issus de l'entraînement
Gaussian Splatting (nerfstudio, méthode splatfacto) de la scène *room* :

| Fichier | Contenu |
|---|---|
| `splat.ply` | Modèle Gaussian Splatting exporté (positions, échelles, rotations, opacités, harmoniques sphériques de degré 3) |
| `dataparser_transforms.json` | Transformation (`transform`, matrice 3×4) et facteur d'échelle (`scale`) appliqués par nerfstudio aux poses COLMAP |

`config.py` lit la transformation et l'échelle dans `dataparser_transforms.json`.
Si le fichier est absent, il utilise les valeurs du notebook, écrites en dur.

Les poses des caméras d'origine ne sont pas incluses : elles proviennent du jeu de
données [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) (scène *room*). Après
téléchargement, placer le dossier `360_v2/room/` dans `donnees/` :

```
donnees/360_v2/room/poses_bounds.npy   poses des caméras (format LLFF), nécessaire à l'entraînement
donnees/360_v2/room/images/            images d'origine, utilisées uniquement par visualize.py
```

Ces positions de caméra, ramenées dans le repère du splat par la transformation
nerfstudio, définissent l'espace de travail : tirage des positions de départ et
des cibles, et bornes des déplacements.

Tous les chemins peuvent être modifiés en ligne de commande (`--ply`,
`--poses-bounds`, `--images-dir`).

## Expériences

Chaque valeur de `--exp` reprend une section du notebook, avec ses hyperparamètres.

| `--exp` | Environnement | État | Action | Algorithmes |
|---|---|---|---|---|
| `yaw` | `VisualServoingEnvYaw` | poses courante et cible, yaw (8) | 3 translations + yaw | A3C, PPO |
| `euler` | `VisualServoingEnvEuler` | poses + yaw/pitch/roll (12) | 3 translations + 3 rotations | A3C, PPO |
| `orientation` | `OrientationOnlyEnv` | idem (12), position figée sur la cible | 3 rotations | PPO |
| `orientation_decoupled` | `OrientationOnlyEnvDecoupled` | idem | 3 rotations | PPO (+ StepLR, `max_std=0.15`) |
| `quat_toy` | `SimpleVisualServoingEnv` | erreur de position dans le repère caméra + quaternion relatif (7) | translation locale + rotvec | PPO + évaluation |
| `quat` | `QuaternionVisualServoingEnv` | idem, dans l'espace de travail réel | idem | A3C, PPO |
| `resnet` | `GSplatVisualServoingEnv` | z\* − z, features ResNet-18 normalisées (512) | 3 translations + 3 rotations | A3C, PPO + trajectoire |

**Récompense.** Dans tous les environnements, la récompense est une récompense de
progrès (diminution de la distance à la cible entre deux pas, multipliée par un
facteur), plus un bonus lorsque la cible est atteinte.

**Distances utilisées.**
- `yaw` : distance position normalisée par la diagonale de l'espace de travail + écart de yaw / 180 ;
- `euler` : distance position normalisée + distance géodésique sur SO(3) / 180 ;
- `orientation` : distance géodésique seule ; `orientation_decoupled` : moyenne des écarts yaw, pitch, roll ;
- `quat_toy`, `quat` : distance euclidienne en position + angle de la rotation relative (rad) ;
- `resnet` : distance cosinus entre features × 100.

**Cible.** Une pose cible est tirée au début du script et reste fixe pendant tout
l'entraînement. Comme dans le notebook, seule la **position** cible est fixée pour
`yaw`, `euler`, `orientation*` et `quat` (l'orientation cible est retirée à chaque
épisode) ; pour `resnet`, la pose complète est fixée.

## Utilisation

```bash
pip install -r requirements.txt

python train.py --exp euler --algo both
python train.py --exp orientation_decoupled
python train.py --exp resnet --algo ppo

python evaluate.py --checkpoint outputs/quat_toy_ppo.pt
python visualize.py dataset
python visualize.py render --pose 0.1 0.0 0.2 --yaw 90
```

Chaque run écrit dans `outputs/` le checkpoint (`<exp>_<algo>.pt`, qui contient aussi
la cible), l'historique des récompenses et distances (`<exp>_<algo>_history.npz`) et
les figures. Le rendu gsplat nécessite un GPU CUDA ; seule l'expérience `resnet`
en a besoin pendant l'entraînement (pour `yaw` et `euler`, il sert uniquement à
l'aperçu de la cible, désactivable avec `--no-preview`).

## Remarque

- Les figures sont sauvegardées en PNG (et affichées seulement avec `--show`).
