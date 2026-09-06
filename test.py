import torch

from brain import Brain
from megaVecto import MegaCrea

SUB_STEP = 10
dt = torch.tensor(1.0 / SUB_STEP)
device = torch.device("cpu")

CHEMIN = "elite_mutant/champion_gen_0_score_25.9_family_14.pt"
NB_SEEDS = 50
NB_FRAMES = 1000

# ---------- chargement ----------
donnees = torch.load(CHEMIN, map_location=device)

x = donnees['x']
y = donnees['y']
is_bone = donnees['is_bone']
muscle1 = donnees['muscle1'].tolist() if torch.is_tensor(donnees['muscle1']) else donnees['muscle1']
muscle2 = donnees['muscle2'].tolist() if torch.is_tensor(donnees['muscle2']) else donnees['muscle2']
target_length = donnees['target_length']
max_noeuds = donnees['max_noeuds']
max_muscles = donnees['max_muscles']

num_nodes_reel = len(x)
num_muscles_reel = len(muscle1)


def pad_1d(vec, size, dtype=torch.float32, pad_value=0.0):
    t = torch.tensor(vec, dtype=dtype)
    if len(vec) < size:
        t = torch.cat([t, torch.full((size - len(vec),), pad_value, dtype=dtype)])
    return t.unsqueeze(0)


dico = {
    "X": pad_1d(x, max_noeuds),
    "Y": pad_1d(y, max_noeuds),
    "m1": pad_1d(muscle1, max_muscles, dtype=torch.long, pad_value=0),
    "m2": pad_1d(muscle2, max_muscles, dtype=torch.long, pad_value=0),
    "stiffness": pad_1d([1.0 + 4 * s for s in is_bone], max_muscles),
    "is_bone": pad_1d(is_bone, max_muscles),
    "base_length": pad_1d(target_length.tolist(), max_muscles),
    "masque_noeuds": (torch.arange(max_noeuds) < num_nodes_reel).float().unsqueeze(0),
    "masque_muscles": (torch.arange(max_muscles) < num_muscles_reel).float().unsqueeze(0),
}

obs_size = max_noeuds * 4 + max_muscles + 1
action_size = max_muscles

cerveau = Brain(obs_size, action_size)
cerveau.load_state_dict(donnees['brain_weights'])
cerveau.eval()

print(f"{CHEMIN}")
print(f"{num_nodes_reel} noeuds, {num_muscles_reel} liens\n")

# ---------- balayage des seeds ----------
resultats = []

with torch.no_grad():
    for seed in range(NB_SEEDS):
        torch.manual_seed(seed)
        c = MegaCrea(dico, batch_size=1, device=device)

        depart = c.X[0, 0, :num_nodes_reel].mean().item()
        actions = []

        for frame in range(NB_FRAMES):
            if frame % 5 == 0:
                obs = c.get_observation(frame).reshape(1, obs_size)
                action = cerveau(obs).reshape(1, 1, action_size)
                if frame % 100 == 0:
                                print(frame, action.flatten()[:4].tolist())
                bruit = torch.randn_like(action) * 0.02
                action+=bruit
                actions.append(action.abs().mean().item())
                c.apply_action(action, frame)
            for _ in range(SUB_STEP):
                c.apply_physics(dt)
            
            

        fin = c.X[0, 0, :num_nodes_reel].mean().item()
        dist = fin - depart
        amp = sum(actions) / len(actions)
        resultats.append((dist, seed, amp))
        print(f"seed {seed:3d} | deplacement {dist:9.1f} | |action| moy {amp:.3f}")

resultats.sort(reverse=True)
print("\n--- top 5 ---")
for dist, seed, amp in resultats[:5]:
    print(f"seed {seed:3d} | deplacement {dist:9.1f} | |action| moy {amp:.3f}")