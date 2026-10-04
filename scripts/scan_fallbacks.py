"""Print output lines that mention a fallback, a skipped step or a missing package."""
import json, re, sys
PAT = re.compile(r"not installed|fallback|fall back|skipped|unavailable|did not run|not available|NotEnoughMemory|offline", re.I)
for f in sys.argv[1:]:
    nb = json.load(open(f)); print("==", f)
    for i, c in enumerate(nb["cells"]):
        for o in c.get("outputs", []):
            t = "".join(o.get("text", "")) if o["output_type"] == "stream" else ""
            for l in t.splitlines():
                if PAT.search(l): print(f"  [{i}] {l[:220]}")
