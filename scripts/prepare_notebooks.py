"""Build the assignment notebooks from the reference Colabs in notebooks/.

For each part this script: clears every reference output (so the checked-in outputs
come only from our own run), drops Colab widget-state metadata (GitHub cannot render
it), replaces the install cells with Colab-guarded installs, adds an assignment header
and a run-environment cell, and points the Open-in-Colab badge at this repository.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks"
REPO = "anita-agasaveeran/Assignment3"

AG_INSTALL = '''# ============================================================
#  Install (Colab only) - the local run uses the environment in requirements/autogluon.txt
# ============================================================
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !pip install -q --upgrade pip setuptools wheel
    !pip install -q autogluon
    print("AutoGluon installed. First run only: Runtime > Restart session, then continue from the next cell.")
else:
    print("Not on Colab: using the pre-built local environment, nothing to install.")'''

PC_INSTALL = '''# ============================================================
#  Install (Colab only) - the local run uses the environment in requirements/pycaret.txt
# ============================================================
# PyCaret 3.3.2 supports Python 3.9-3.11 only. On Colab pick the Python 3.11 image:
#   Runtime > Change runtime type > Runtime version > 2025.07 > Save
# pip may print a red "dependency resolver" block about google-colab / plotnine pins: it is harmless here.
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    assert sys.version_info[:2] == (3, 11), "Need Python 3.11, got " + sys.version.split()[0] + ". Set Runtime version to 2025.07."
    !pip install -q --only-binary=:all: "pycaret[analysis,models]==3.3.2" fastapi uvicorn
    print("pip finished on Python", sys.version.split()[0], "- first run only: Runtime > Restart session, then continue.")
else:
    assert sys.version_info[:2] <= (3, 11), "PyCaret 3.3.2 needs Python <= 3.11"
    print("Not on Colab: using the pre-built local environment (Python", sys.version.split()[0] + "), nothing to install.")'''

RAPIDS_INSTALL = '''# ============================================================
#  RAPIDS check (Colab GPU runtime: Runtime > Change runtime type > T4 GPU)
# ============================================================
# Colab GPU images ship cuDF / cuML / cuGraph pre-installed. If an import fails, uncomment the pip line,
# run it, then Runtime > Restart session.
# !pip install -q cudf-cu12 cuml-cu12 cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
import importlib
for pkg in ["cudf", "cuml", "cugraph", "cupy", "xgboost"]:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg:8s} {getattr(m, '__version__', '?')}")
    except Exception as e:  # noqa: BLE001
        print(f"{pkg:8s} NOT available ({type(e).__name__}) - the notebook falls back to the CPU path")
!nvidia-smi || echo "no NVIDIA GPU visible"'''

ENV_CELL = '''# ============================================================
#  Run environment: proves where and when this copy of the notebook was executed
# ============================================================
import sys, platform, datetime, os
print("Run by       : Anita Agasaveeran (CMPE 255, Assignment 3)")
print("Executed at  :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("Runtime      :", "Google Colab" if "google.colab" in sys.modules else "local Jupyter kernel")
print("Python       :", sys.version.split()[0], "|", platform.platform())
print("CPU cores    :", os.cpu_count())
try:
    import torch
    print("CUDA GPU     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
          "| Apple MPS:", getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
except Exception:  # noqa: BLE001
    print("torch        : not installed")'''

PARTS = [
    dict(src="final_kmeans_zero_to_hero.ipynb", dst="part1_kmeans/kmeans_and_variations.ipynb", part=1,
         title="K-means and its variations",
         ref="https://colab.research.google.com/drive/1Ye2fCkgzbqhpZr7g5Z_DTSoAufkYW3jP",
         install=None, env_after="All libraries imported successfully!"),
    dict(src="final_autogluon_capabilities_tour.ipynb", dst="part2_autogluon_capabilities/autogluon_capabilities_tour.ipynb", part=2,
         title="AutoGluon - landscape of capabilities",
         ref="https://colab.research.google.com/drive/12u1wvSBmeFr7wGVodVafpJJuE7bq35tM",
         install=AG_INSTALL, env_after="All libraries imported successfully!",
         fixes=[  # Mitra's 7 GB CPU estimate trips AutoGluon's memory guard once the kernel holds the earlier predictors;
                  # on a laptop with swap it fits, so relax the guard locally and keep the default on Colab's smaller VM.
             ('FM_HP = {"fine_tune": False} if FM_KEY == "MITRA" else {}   # Mitra defaults to fine-tuning, which wants a GPU; zero-shot here',
              'FM_HP = {"fine_tune": False} if FM_KEY == "MITRA" else {}   # Mitra defaults to fine-tuning, which wants a GPU; zero-shot here\n'
              'if "google.colab" not in sys.modules:\n'
              '    FM_HP["ag.max_memory_usage_ratio"] = 1.5                   # local run: allow the CPU memory estimate past the 90% guard')]),
    dict(src="final_autogluon_zero_to_hero.ipynb", dst="part3_autogluon_end2end/autogluon_end_to_end.ipynb", part=3,
         title="AutoGluon - end-to-end ML with metrics",
         ref="https://colab.research.google.com/drive/1rMUbJIsFg9IvsLxcpqSyfuULlCkmBYQq",
         install=AG_INSTALL, env_after="def make_budget",
         fixes=[  # ag_args_fit num_cpus must not exceed the fit's num_cpus cap (4): os.cpu_count() is 14 on the M4 Pro
             ("SAFE_N_CPUS = os.cpu_count() or 1", "SAFE_N_CPUS = min(N_CPUS, os.cpu_count() or 1)"),
             # prajjwal1/bert-tiny's config.json has no model_type, so current transformers refuses it and the
             # multimodal Part silently fell back to TF-IDF; Google's official copy of the same 2-layer BERT loads.
             ("prajjwal1/bert-tiny", "google/bert_uncased_L-2_H-128_A-2")]),
    dict(src="final_nvidia_rapids_zero_to_hero.ipynb", dst="part4_rapids_vs_cpu/rapids_vs_cpu.ipynb", part=4,
         title="NVIDIA RAPIDS (GPU) compared with the CPU version",
         ref="https://colab.research.google.com/drive/1Yb76g6biaSAWUNzToceC9G2QkZCAMnS3",
         install=RAPIDS_INSTALL, env_after="def make_budget",
         fixes=[  # to_host(): pandas objects also have .get(key); only CuPy arrays should take the .get() branch
             ('(obj.get() if hasattr(obj, "get") else obj)',
              '(obj.get() if hasattr(obj, "get") and not isinstance(obj, (pd.DataFrame, pd.Series)) else obj)')]),
    dict(src="final_pycaret_capabilities_tour.ipynb", dst="part5_pycaret_capabilities/pycaret_capabilities_tour.ipynb", part=5,
         title="PyCaret - landscape of capabilities",
         ref="https://colab.research.google.com/drive/1OlqgjP5B45FvR4t6nzyLAGIedxpFMj8W",
         install=PC_INSTALL, env_after="All libraries imported successfully!"),
    dict(src="final_pycaret_zero_to_hero.ipynb", dst="part6_pycaret_mlops/pycaret_mlops.ipynb", part=6,
         title="PyCaret - end-to-end and MLOps",
         ref="https://colab.research.google.com/drive/1EIPQdCHnyvtj7WdyZh1SBbR7ruxEvUkg",
         install=PC_INSTALL, env_after="def make_budget"),
]


# Bugs found while executing the reference notebooks, fixed in every part where they occur.
COMMON_FIXES = [
    # pretty(): "{:.3f}" was applied to text columns too -> ValueError on any frame with a string column
    ('sty = df.style.format(fmt, na_rep="-")',
     'sty = df.style.format(fmt, na_rep="-", subset=df.select_dtypes("number").columns)'),
    # exceptions with an empty message (e.g. NotEnoughMemoryError) crashed the guarded fallback's own print
    ('str(e).splitlines()[0]', "(str(e) or '-').splitlines()[0]"),
]


def cell(kind, text):
    c = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def is_install(c):
    s = "".join(c["source"])
    live = [l for l in s.splitlines() if l.strip() and not l.strip().startswith("#")]
    return c["cell_type"] == "code" and any(re.match(r"\s*[!%].*(pip|uv) .*install", l) for l in live) \
        and len(live) <= 8


def header(p):
    path = p["dst"]
    return f'''# CMPE 255 · Assignment 3 · Part {p["part"]}: {p["title"]}

**Student:** Anita Agasaveeran · San José State University · Fall 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/{path})

| | |
|---|---|
| Reference Colab | [{p["ref"]}]({p["ref"]}) |
| This notebook | my own executed copy - every output below comes from my run (see the *Run environment* cell) |
| How to run | Open in Colab and *Runtime > Run all* (see the install cell for any restart note), or locally with the environment in `requirements/` |

---'''


def build(p):
    nb = json.loads((SRC / p["src"]).read_text())
    nb["metadata"].pop("widgets", None)
    nb["metadata"]["language_info"] = {"name": "python"}
    if p["part"] == 4:
        nb["metadata"]["colab"] = {"provenance": [], "gpuType": "T4"}
        nb["metadata"]["accelerator"] = "GPU"
    cells = []
    installed = False
    for c in nb["cells"]:
        if not "".join(c["source"]).strip():
            continue                                   # empty scratch cells from the reference
        c.pop("id", None)
        c.get("metadata", {}).pop("colab", None); c.get("metadata", {}).pop("outputId", None)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            if is_install(c):
                if p["install"] and not installed:
                    cells.append(cell("code", p["install"])); installed = True
                continue                               # drop the reference's raw pip cells
        src = "".join(c["source"])
        src = re.sub(r"https://colab\.research\.google\.com/github/dlmastery/class/blob/main/colabs/\S+?\.ipynb",
                     f"https://colab.research.google.com/github/{REPO}/blob/main/{p['dst']}", src)
        src = src.replace('model="claude-sonnet-5"', 'model="claude-sonnet-5-5"')
        for old, new in COMMON_FIXES + p.get("fixes", []):
            src = src.replace(old, new)
        c["source"] = src.splitlines(keepends=True)
        cells.append(c)
        if c["cell_type"] == "code" and p["env_after"] in src and not any("Run environment" in "".join(x["source"]) for x in cells):
            cells.append(cell("code", ENV_CELL))
    if p["install"] and not installed:                 # reference had no install cell: put ours right after the title
        cells.insert(1, cell("code", p["install"]))
    assert any("Run environment" in "".join(x["source"]) for x in cells), p["dst"]
    cells.insert(0, cell("markdown", header(p)))
    nb["cells"] = cells
    nb["nbformat"], nb["nbformat_minor"] = 4, 5
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"p{p['part']}c{i:03d}"
    out = ROOT / p["dst"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"{out.relative_to(ROOT)}: {len(cells)} cells")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for p in PARTS:
        if not want or str(p["part"]) in want:
            build(p)
