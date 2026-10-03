"""
Convertit une vidéo produite par visualize.py en GIF léger pour le README.

Usage :
    python tools/mp4_to_gif.py videos/ppo_seed1.mp4 docs/swim_ppo.gif --largeur 600 --pas 2
"""

import argparse

import cv2
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("sortie")
    p.add_argument("--largeur", type=int, default=600, help="largeur du GIF en pixels")
    p.add_argument("--pas", type=int, default=2, help="garde une image sur N")
    p.add_argument("--fps", type=float, default=30.0, help="images par seconde du GIF")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.video)
    images, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % a.pas == 0:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (a.largeur, int(h * a.largeur / w)), interpolation=cv2.INTER_AREA)
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not images:
        raise SystemExit(f"aucune image lue dans {a.video}")
    images[0].save(a.sortie, save_all=True, append_images=images[1:],
                   duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"🎞  {a.sortie} ({len(images)} images)")


if __name__ == "__main__":
    main()
