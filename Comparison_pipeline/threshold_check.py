# threshold_check.py — run this once to find the right threshold
import config
from common import load_image, detect_faces, iter_dataset

scores_by_identity = {}
for identity, image_path in iter_dataset(config.DATA_ROOT):
    try:
        img = load_image(image_path)
        faces = detect_faces(img)
        if faces:
            best_score = faces[0].det_score  # highest score face in this image
            scores_by_identity.setdefault(identity, []).append(round(best_score, 3))
    except Exception as e:
        print(f"Error {image_path}: {e}")

print(f"\n{'Identity':<20} {'Min':>6} {'Max':>6} {'Avg':>6} {'<0.85':>6} {'Total':>6}")
print("-" * 60)
for identity, scores in sorted(scores_by_identity.items()):
    below = sum(1 for s in scores if s < 0.85)
    print(f"{identity:<20} {min(scores):>6.3f} {max(scores):>6.3f} "
          f"{sum(scores)/len(scores):>6.3f} {below:>6} {len(scores):>6}")