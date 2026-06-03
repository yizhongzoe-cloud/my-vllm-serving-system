#!/usr/bin/env python3
# Replace interior ASCII double-quotes inside the Chinese note arg (the line
# right after `notes(s,`) with full-width corner brackets, so they don't clash
# with the Python string delimiters.
path = "build_slides.py"
lines = open(path, encoding="utf-8").read().split("\n")
out = []
prev_notes = False
for ln in lines:
    s = ln.strip()
    if prev_notes and s.startswith('"'):
        i = ln.index('"'); j = ln.rindex('"')
        mid = ln[i + 1:j]
        res = []; open_q = True
        for ch in mid:
            if ch == '"':
                res.append("「" if open_q else "」"); open_q = not open_q
            else:
                res.append(ch)
        ln = ln[:i + 1] + "".join(res) + ln[j:]
        prev_notes = False
    elif s.endswith("notes(s,"):
        prev_notes = True
    else:
        prev_notes = False
    out.append(ln)
open(path, "w", encoding="utf-8").write("\n".join(out))
print("fixed interior quotes in Chinese notes")
