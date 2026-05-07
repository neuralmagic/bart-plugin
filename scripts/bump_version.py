#!/usr/bin/env python3
"""Simple script to bump the package version in setup.py and pyproject.toml."""

import argparse
import re
import sys
from pathlib import Path


def get_current_version(content: str) -> str | None:
    """Extract version from file content."""
    match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
    return match.group(1) if match else None


def bump_version(version: str, part: str) -> str:
    """Bump the specified part of the version."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Expected semantic version (x.y.z), got: {version}")

    major, minor, patch = map(int, parts)

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown version part: {part}")

    return f"{major}.{minor}.{patch}"


def update_file(filepath: Path, old_version: str, new_version: str) -> bool:
    """Update version in a file. Returns True if file was modified."""
    if not filepath.exists():
        return False

    content = filepath.read_text()
    new_content = re.sub(
        rf'(version\s*=\s*["\']){re.escape(old_version)}(["\'])',
        rf"\g<1>{new_version}\g<2>",
        content,
    )

    if content != new_content:
        filepath.write_text(new_content)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Bump package version")
    parser.add_argument(
        "part",
        choices=["major", "minor", "patch"],
        nargs="?",
        default="patch",
        help="Version part to bump (default: patch)",
    )
    parser.add_argument(
        "--set",
        dest="set_version",
        metavar="VERSION",
        help="Set a specific version instead of bumping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    files = [root / "setup.py", root / "pyproject.toml"]

    # Get current version from pyproject.toml
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        print("Error: pyproject.toml not found", file=sys.stderr)
        sys.exit(1)

    current_version = get_current_version(pyproject.read_text())
    if not current_version:
        print("Error: Could not find version in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    # Determine new version
    if args.set_version:
        new_version = args.set_version
    else:
        new_version = bump_version(current_version, args.part)

    print(f"Version: {current_version} -> {new_version}")

    if args.dry_run:
        print("Dry run - no files modified")
        return

    # Update files
    for filepath in files:
        if update_file(filepath, current_version, new_version):
            print(f"Updated: {filepath.name}")
        else:
            print(f"Skipped: {filepath.name} (not found or no changes)")

    print(f"\nDone! Don't forget to commit and tag: git tag v{new_version}")


if __name__ == "__main__":
    main()
