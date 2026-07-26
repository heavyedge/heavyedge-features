#!/usr/bin/env python3
"""Print the body of a version section from a Keep a Changelog file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VERSION_HEADING = re.compile(r"^##\s+(?P<title>.+?)\s*$", re.MULTILINE)


def heading_version(title: str) -> str:
    """Return the version portion of a level-two changelog heading."""
    title = title.strip()
    if title.startswith("["):
        closing_bracket = title.find("]")
        if closing_bracket != -1:
            return title[1:closing_bracket].strip()

    # Supports both "## v1.2.3" and "## v1.2.3 - 2026-07-20".
    return re.split(r"\s+-\s+", title, maxsplit=1)[0].strip()


def extract_section(changelog: str, version: str) -> str:
    headings = list(VERSION_HEADING.finditer(changelog))
    for index, heading in enumerate(headings):
        if heading_version(heading.group("title")) != version:
            continue

        end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        )
        return changelog[heading.end() : end].strip()

    raise ValueError(f"No CHANGELOG.md section found for release tag '{version}'.")


def extract_sections(changelog: str) -> list[dict[str, str]]:
    """Return every non-empty version section in the changelog."""
    headings = list(VERSION_HEADING.finditer(changelog))
    sections = []

    for index, heading in enumerate(headings):
        end = (
            headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        )
        body = changelog[heading.end() : end].strip()
        if body:
            sections.append(
                {"version": heading_version(heading.group("title")), "body": body}
            )

    return sections


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", help="Release tag to look up")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print all non-empty version sections as JSON",
    )
    parser.add_argument("--file", type=Path, default=Path("CHANGELOG.md"))
    args = parser.parse_args()

    try:
        changelog = args.file.read_text(encoding="utf-8")
        if args.all:
            print(json.dumps(extract_sections(changelog), ensure_ascii=False))
            return 0

        if not args.version:
            parser.error("version is required unless --all is used")

        section = extract_section(changelog, args.version)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    if not section:
        print(
            f"CHANGELOG.md section for release tag '{args.version}' is empty.",
            file=sys.stderr,
        )
        return 1

    print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
