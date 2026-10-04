"""List cells whose outputs contain an error, with the exception line."""
import json, sys
for f in sys.argv[1:]:
    nb = json.load(open(f))
    errs = [(i, o) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error"]
    unexec = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("execution_count") is None)
    print(f"== {f}: {len(errs)} error cells, {unexec} unexecuted code cells")
    for i, o in errs:
        print(f"  [{i}] {o['ename']}: {o['evalue'][:300]}")
