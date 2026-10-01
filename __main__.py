"""
Entry point for `python -m my_design_final <subcommand>`.

Subcommands:
    train       — Offline training of generator + verifier (CPU)
    infer       — Online inference (seed_expand / target mode)
"""
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m my_design_final {train|infer} [options]")
        sys.exit(1)

    cmd = sys.argv[1]
    sys.argv = sys.argv[1:]  # shift so argparse sees correct argv

    if cmd == "train":
        from .train import main as train_main
        train_main()
    elif cmd == "infer":
        from .infer import main as infer_main
        infer_main()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m my_design_final {train|infer} [options]")
        sys.exit(1)

if __name__ == "__main__":
    main()
