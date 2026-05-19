from pathlib import Path
import importlib.util

_mt_spec = importlib.util.spec_from_file_location("mt", Path(__file__).with_name("04_model_training_testing.py"))
mt = importlib.util.module_from_spec(_mt_spec)
assert _mt_spec and _mt_spec.loader
_mt_spec.loader.exec_module(mt)

base_dir = Path(__file__).resolve().parent
candidate_paths = [base_dir / "models" / "fus_full_bundle_gray.pt"]

bundle_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
bundle = mt.load_bundle(bundle_path)

print(bundle.keys())
print(bundle["targets"])
print(bundle["output_alphas"])
print(bundle["model"])