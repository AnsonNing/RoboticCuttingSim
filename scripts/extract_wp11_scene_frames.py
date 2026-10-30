"""Extract first/last presentation close-up frames from recorded live WP-11 episodes (read-only)."""
import glob, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
runs = {
 "trim_rho10_clear": "artifacts/wp11/phaseC5/live_trim_rho10_clear_002/episode_000/observations",
 "test_b_s1502": "artifacts/wp11/phaseE/round1_live_001/E1_TEST_B_s1502/run_001/episode_000/observations",
 "test_c_s1602": "artifacts/wp11/phaseE/round1_live_003/E1_TEST_C_s1602/run_001/episode_000/observations",
 "slice_push_relocate_s1301": "artifacts/wp11/phaseE/round1_live_004/E1_SLICE_PUSH_s1301/run_001/episode_000/observations",
}
rec = {}
for name, d in runs.items():
    files = sorted(glob.glob(f"{d}/closeup_rgb_*.npz"))
    for tag, f in (("start", files[0]), ("end", files[-1])):
        z = np.load(f)
        img = Image.fromarray(z["rgb"]).resize((640, 360), Image.LANCZOS)
        p = out / f"{name}_{tag}.png"; img.save(p, optimize=True)
        rec[p.name] = {"source": f, "physics_timestamp_s": float(z["physics_timestamp_s"])}
json.dump(rec, open(out / "provenance.json", "w"), indent=1)
print(json.dumps(rec, indent=1))
