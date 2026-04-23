"""CLI entry point."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Grain growth Potts simulation")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--q", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
