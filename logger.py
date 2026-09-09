"""
logger.py — logger structuré réutilisable pour train.py / train2.py.

Écrit chaque enregistrement (fin d'épisode / génération) en :
  - JSON Lines (.jsonl)  : un objet JSON par ligne, append-only, jamais corrompu
                           par un crash au milieu d'un run
  - CSV (.csv)           : mêmes champs, pour ouverture directe dans un tableur / pandas

Usage minimal :

    from logger import Logger

    logger = Logger("runs/train_phase1.jsonl", run_config={"lr": lr, "seed": SEED})

    ...
    logger.log(generation=generation, episode=episode,
               score_max=meilleur, score_mean=moyenne_episode,
               grad_norm=float(norm))

Le premier appel à log() fixe les colonnes du CSV (union des clés vues).
Si un appel ultérieur introduit une nouvelle clé, le logger réécrit l'en-tête
du CSV avec la colonne en plus (les lignes déjà écrites gardent un champ vide
pour cette colonne) — pas besoin de connaître le schéma à l'avance.
"""

import csv
import json
import os
import time


class Logger:
    def __init__(self, path, run_config=None, flush_every=1):
        """
        path        : chemin du fichier .jsonl (le .csv est dérivé en changeant
                       l'extension — ex. "runs/foo.jsonl" -> "runs/foo.csv")
        run_config  : dict optionnel de méta-données du run (lr, seed, batch_size, ...),
                       écrit une seule fois en tête du .jsonl (ligne {"_run_config": {...}})
        flush_every : nombre d'appels à log() entre deux flush() disque (1 = flush à chaque
                       appel, le plus sûr en cas de crash ; augmenter si l'I/O devient un goulot)
        """
        self.jsonl_path = path
        base, _ext = os.path.splitext(path)
        self.csv_path = base + ".csv"

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        self._flush_every = max(1, flush_every)
        self._calls_since_flush = 0
        self._fieldnames = []  # ordre d'apparition des colonnes CSV
        self._start_time = time.time()

        # Reprise sur un fichier existant : on relit les colonnes déjà présentes
        # pour ne pas casser le CSV si le run est relancé sur le même chemin.
        file_exists = os.path.isfile(self.csv_path) and os.path.getsize(self.csv_path) > 0
        if file_exists:
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                try:
                    self._fieldnames = next(reader)
                except StopIteration:
                    self._fieldnames = []

        self._jsonl_file = open(self.jsonl_path, "a", encoding="utf-8")

        if run_config is not None:
            record = {"_run_config": run_config, "_wall_time": self._start_time}
            self._jsonl_file.write(json.dumps(record, default=str) + "\n")
            self._jsonl_file.flush()

    def log(self, **fields):
        """Enregistre un point (typiquement : une fin d'épisode ou de génération).

        Accepte n'importe quels champs nommés, ex. :
            logger.log(generation=3, episode=10, score_max=34.2, score_mean=9.1,
                        grad_norm=870.0, explosion=False)

        Ajoute automatiquement 'elapsed_s' (secondes depuis la création du logger)
        si non fourni.
        """
        if "elapsed_s" not in fields:
            fields["elapsed_s"] = round(time.time() - self._start_time, 2)

        # --- JSONL : toujours complet, aucune perte de champ ---
        self._jsonl_file.write(json.dumps(fields, default=str) + "\n")

        # --- CSV : union progressive des colonnes ---
        new_keys = [k for k in fields.keys() if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._rewrite_csv_header()

        row_exists = os.path.isfile(self.csv_path) and os.path.getsize(self.csv_path) > 0
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if not row_exists:
                writer.writeheader()
            writer.writerow(fields)

        self._calls_since_flush += 1
        if self._calls_since_flush >= self._flush_every:
            self._jsonl_file.flush()
            os.fsync(self._jsonl_file.fileno())
            self._calls_since_flush = 0

    def _rewrite_csv_header(self):
        """Réécrit le CSV avec le jeu de colonnes élargi, en conservant les lignes déjà écrites
        (les colonnes manquantes sur les anciennes lignes restent vides)."""
        if not (os.path.isfile(self.csv_path) and os.path.getsize(self.csv_path) > 0):
            return  # rien à réécrire, le prochain writerow posera le bon header

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in old_rows:
                writer.writerow(row)

    def close(self):
        self._jsonl_file.flush()
        self._jsonl_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
