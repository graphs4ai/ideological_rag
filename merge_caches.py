import argparse
import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two cache .pkl files into a single output.")
    parser.add_argument("--old", default="cache_respostas_antigo.pkl", help="Old cache file")
    parser.add_argument("--new", default="cache_respostas.pkl", help="New cache file")
    parser.add_argument("--out", default="cache.pkl", help="Output merged cache file")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    out_path = Path(args.out)

    if not old_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {old_path}")
    if not new_path.exists():
        raise SystemExit(f"Arquivo não encontrado: {new_path}")

    old_cache = load_pickle(old_path)
    new_cache = load_pickle(new_path)

    if not isinstance(old_cache, dict):
        raise SystemExit(f"Formato inesperado em {old_path}: esperado dict, veio {type(old_cache).__name__}")
    if not isinstance(new_cache, dict):
        raise SystemExit(f"Formato inesperado em {new_path}: esperado dict, veio {type(new_cache).__name__}")

    merged = dict(old_cache)

    conflicts = 0
    for k, v in new_cache.items():
        if k in merged and merged[k] != v:
            conflicts += 1
        merged[k] = v  # new wins

    save_pickle(merged, out_path)

    print(
        "OK: merged caches\n"
        f"- old: {old_path} ({len(old_cache)} chaves)\n"
        f"- new: {new_path} ({len(new_cache)} chaves)\n"
        f"- out: {out_path} ({len(merged)} chaves)\n"
        f"- conflitos (new sobrescreveu old): {conflicts}"
    )


if __name__ == "__main__":
    main()
