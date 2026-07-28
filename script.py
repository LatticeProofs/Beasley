#!/usr/bin/env python3
"""Strip every comment from this repo's sources, in place.

    python strip_comments.py            # do it
    python strip_comments.py --dry-run  # preview only

Handles .rs (// /// //! /* */ incl. nested) and .toml (#).
String / raw-string / char literals are preserved verbatim; 'a lifetimes are
not mistaken for char literals.  Skips target/ and .git/.
"""
import sys
import pathlib

MARK = "\x00"   # "a comment was here" -> tidy() drops the line if it is now empty


# --------------------------------------------------------------------- Rust

def strip_rust(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]

        # raw string:  r"..."  r#"..."#  br##"..."##
        if c in "rb" and _raw_start(src, i):
            j = _skip_raw(src, i)
            out.append(src[i:j])
            i = j
            continue

        # normal / byte string:  "..."  b"..."
        if c == '"' or (c == "b" and i + 1 < n and src[i + 1] == '"'):
            j = i + (2 if c == "b" else 1)
            while j < n:
                if src[j] == "\\":
                    j += 2
                elif src[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            out.append(src[i:j])
            i = j
            continue

        # char literal vs lifetime
        if c == "'":
            if i + 1 < n and src[i + 1] == "\\":           # '\n'  '\u{1F}'
                j = i + 2
                while j < n and src[j] != "'":
                    j += 1
                j += 1
                out.append(src[i:j])
                i = j
                continue
            if i + 2 < n and src[i + 2] == "'":            # 'a'
                out.append(src[i:i + 3])
                i += 3
                continue
            out.append(c)                                  # lifetime 'a
            i += 1
            continue

        # line comment
        if src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
            out.append(MARK)
            continue

        # block comment (nested)
        if src.startswith("/*", i):
            depth, start, i = 1, i, i + 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth += 1
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            out.append(MARK + ("\n" + MARK) * src.count("\n", start, i))
            continue

        out.append(c)
        i += 1
    return "".join(out)


def _raw_start(s: str, i: int) -> bool:
    j = i + 1
    if s[i] == "b":
        if j < len(s) and s[j] == "r":
            j += 1
        else:
            return False
    while j < len(s) and s[j] == "#":
        j += 1
    return j < len(s) and s[j] == '"'


def _skip_raw(s: str, i: int) -> int:
    j = i + 1
    if s[i] == "b":
        j += 1
    hashes = 0
    while s[j] == "#":
        hashes += 1
        j += 1
    j += 1                                    # opening quote
    close = '"' + "#" * hashes
    k = s.find(close, j)
    return len(s) if k < 0 else k + len(close)


# --------------------------------------------------------------------- TOML

def strip_toml(src: str) -> str:
    out = []
    for line in src.split("\n"):
        res, i, n = [], 0, len(line)
        while i < n:
            c = line[i]
            if c in "\"'":                    # quoted value: copy verbatim
                q, j = c, i + 1
                while j < n:
                    if q == '"' and line[j] == "\\":
                        j += 2
                    elif line[j] == q:
                        j += 1
                        break
                    else:
                        j += 1
                res.append(line[i:j])
                i = j
                continue
            if c == "#":
                res.append(MARK)
                break
            res.append(c)
            i += 1
        out.append("".join(res))
    return "\n".join(out)


# --------------------------------------------------------------------- shared

def tidy(text: str) -> str:
    """Drop lines that held nothing but a comment; collapse blank-line runs."""
    lines = []
    for ln in text.split("\n"):
        bare = ln.replace(MARK, "").rstrip()
        if MARK in ln and bare == "":
            continue
        lines.append(bare)
    out, blank = [], False
    for ln in lines:
        if ln == "":
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(ln)
    while out and out[0] == "":
        out.pop(0)
    return "\n".join(out).rstrip() + "\n"


SKIP = {"target", ".git", ".github"}
HANDLERS = {".rs": strip_rust, ".toml": strip_toml}


def main() -> int:
    dry = "--dry-run" in sys.argv
    root = pathlib.Path(".").resolve()

    if not (root / "Cargo.toml").is_file():
        print(f"error: no Cargo.toml in {root} -- run this from the repo root", file=sys.stderr)
        return 1

    total_before = total_after = touched = 0
    for p in sorted(root.rglob("*")):
        if p.is_dir() or p.suffix not in HANDLERS:
            continue
        if SKIP & set(p.relative_to(root).parts):
            continue
        before = p.read_text(encoding="utf-8")
        after = tidy(HANDLERS[p.suffix](before))
        total_before += len(before)
        total_after += len(after)
        if after != before:
            touched += 1
            print(f"{p.relative_to(root)}: {len(before)} -> {len(after)} bytes")
            if not dry:
                p.write_text(after, encoding="utf-8", newline="\n")

    verb = "would change" if dry else "changed"
    saved = total_before - total_after
    pct = saved / total_before * 100 if total_before else 0
    print(f"\n{verb} {touched} files, {total_before} -> {total_after} bytes ({saved} removed, {pct:.1f}%)")
    if not dry:
        print("now run:  cargo test --release")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
