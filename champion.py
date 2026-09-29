"""
champion.py — chargement d'un champion .pt, partagé par train2.py et baselines/.

Reconstruit le dico "méga-univers" pour UNE créature (POP_SIZE = 1), avec
exactement le même padding qu'à l'entraînement d'origine.
"""

from dataclasses import dataclass
from typing import Optional

import torch


def pad_1d(vec, size, device, dtype=torch.float32, pad_value=0.0):
    """Pad un vecteur 1D à la taille voulue et ajoute la dimension POP_SIZE=1."""
    t = torch.tensor(vec, dtype=dtype)
    if len(vec) < size:
        t = torch.cat([t, torch.full((size - len(vec),), pad_value, dtype=dtype)])
    return t.unsqueeze(0).to(device)


@dataclass
class Champion:
    x: list
    y: list
    is_bone: list
    muscle1: list
    muscle2: list
    target_length: torch.Tensor
    max_noeuds: int
    max_muscles: int
    brain_weights: Optional[dict]
    dico: dict

    @property
    def num_nodes(self) -> int:
        return len(self.x)

    @property
    def num_muscles(self) -> int:
        return len(self.muscle1)

    @property
    def obs_size(self) -> int:
        return self.max_noeuds * 4 + self.max_muscles + 1

    @property
    def action_size(self) -> int:
        return self.max_muscles

    @property
    def masque_muscles_actifs(self) -> torch.Tensor:
        """[max_muscles] : 1 pour les vrais muscles (ni os, ni fantômes). Seules
        ces sorties du cerveau ont un effet sur la physique."""
        return self.dico["masque_muscles"][0] * (self.dico["is_bone"][0] == 0.0).float()

    def to_save_dict(self, brain_weights) -> dict:
        """Même format que les .pt produits par train.py / train2.py."""
        return {
            'x': self.x, 'y': self.y, 'is_bone': self.is_bone,
            'muscle1': torch.tensor(self.muscle1, dtype=torch.long),
            'muscle2': torch.tensor(self.muscle2, dtype=torch.long),
            'stiffness': torch.tensor([1.0 + 4 * s for s in self.is_bone], dtype=torch.float32),
            'target_length': self.target_length,
            'brain_weights': brain_weights,
            'max_noeuds': self.max_noeuds, 'max_muscles': self.max_muscles,
        }


def charger_champion(chemin, device) -> Champion:
    donnees = torch.load(chemin, map_location=device)

    x = donnees['x']
    y = donnees['y']
    is_bone = donnees['is_bone']
    muscle1 = donnees['muscle1'].tolist() if torch.is_tensor(donnees['muscle1']) else donnees['muscle1']
    muscle2 = donnees['muscle2'].tolist() if torch.is_tensor(donnees['muscle2']) else donnees['muscle2']
    target_length = donnees['target_length']
    max_noeuds = donnees['max_noeuds']
    max_muscles = donnees['max_muscles']
    base_length = target_length.tolist() if torch.is_tensor(target_length) else list(target_length)

    n_noeuds, n_muscles = len(x), len(muscle1)
    dico = {
        "X": pad_1d(x, max_noeuds, device),
        "Y": pad_1d(y, max_noeuds, device),
        "m1": pad_1d(muscle1, max_muscles, device, dtype=torch.long, pad_value=0),
        "m2": pad_1d(muscle2, max_muscles, device, dtype=torch.long, pad_value=0),
        "stiffness": pad_1d([1.0 + 4 * s for s in is_bone], max_muscles, device),
        "is_bone": pad_1d(is_bone, max_muscles, device),
        "base_length": pad_1d(base_length, max_muscles, device),
        "masque_noeuds": (torch.arange(max_noeuds, device=device) < n_noeuds).float().unsqueeze(0),
        "masque_muscles": (torch.arange(max_muscles, device=device) < n_muscles).float().unsqueeze(0),
    }

    return Champion(
        x=x, y=y, is_bone=is_bone, muscle1=muscle1, muscle2=muscle2,
        target_length=target_length, max_noeuds=max_noeuds, max_muscles=max_muscles,
        brain_weights=donnees.get('brain_weights'), dico=dico,
    )
