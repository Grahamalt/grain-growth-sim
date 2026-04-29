"""CLI entry point: run the Phase 3 experiment campaign."""
import argparse
import os

from experiments import run_all


def main():
    parser = argparse.ArgumentParser(description="Grain growth Potts experiment campaign")
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "results"),
        help="Output directory (creates figures/ and data/ subdirs)")
    args = parser.parse_args()
    run_all(args.out_dir)


if __name__ == "__main__":
    main()
