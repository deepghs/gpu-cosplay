#!/usr/bin/env python3
"""CLI wrapper: `gpu-cosplay-apply [-- cmd args...]` inside the container.

If given a command, runs it after applying the VRAM cap (preferring exec).
If not, just prints the cap and exits.
"""
import os
import sys


def main():
    sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
    import gpu_cosplay_inject  # noqa: F401
    gpu_cosplay_inject.apply(verbose=True)
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        return 0
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    sys.exit(main() or 0)
