#!/usr/bin/env python3
"""Build index.html from source fragments.

Usage:
    python3 build.py

Reads template parts from src/, slides from src/slides/ (alphabetical order),
branches from src/branches/ (alphabetical order), and assembles them into index.html.
"""
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
SRC = BASE / 'src'
IFRAME_SRC_RE = re.compile(r'<iframe\b[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)


def read(name):
    path = SRC / name
    return path.read_text(encoding='utf-8')


def read_all(subdir):
    directory = SRC / subdir
    files = sorted(path for path in directory.iterdir() if path.suffix == '.html')
    return '\n'.join(path.read_text(encoding='utf-8') for path in files)


def warn_missing_iframes(html):
    for iframe_src in IFRAME_SRC_RE.findall(html):
        if iframe_src.startswith(('http://', 'https://', '//')):
            continue
        iframe_path = (BASE / iframe_src).resolve()
        if not iframe_path.exists():
            print(f'Warning: iframe source not found: {iframe_src}')


def main():
    top = read('head.html') + '\n' + read('body-top.html')
    slides = read_all('slides')
    mid = read('mid.html')
    branches = read_all('branches')
    bot = read('bot.html')

    mid = mid.replace('<!-- BRANCHES_PLACEHOLDER -->', branches)

    out = top + '\n' + slides + '\n' + mid + '\n' + bot

    warn_missing_iframes(out)
    (BASE / 'index.html').write_text(out, encoding='utf-8')

    print(f'Built index.html: {len(out):,} bytes, {slides.count("<section")} slides, {branches.count("branch-page")} branch pages')


if __name__ == '__main__':
    main()
