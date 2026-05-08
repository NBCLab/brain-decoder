"""Sphinx configuration for braindec documentation."""

import os
import sys

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, ROOT)

project = "braindec"
copyright = "2025, Braindec developers"
author = "Braindec developers"

try:
    from braindec._version import __version__
    release = __version__
except ImportError:
    release = "unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "nilearn": ("https://nilearn.github.io/stable", None),
    "nimare": ("https://nimare.readthedocs.io/en/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# sphinx-gallery configuration.
# Notebooks for Colab are generated separately via docs/make_notebooks.py
# (which uses jupytext and injects the braindec install cell).  sphinx-gallery
# is used only to build the HTML gallery pages from the .py sources.
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    # Only process files whose names start with two digits.
    "filename_pattern": r"/\d{2}_",
    # Set to True locally to execute examples and capture outputs.
    # On ReadTheDocs the examples are not executed (too slow / GPU-dependent).
    "plot_gallery": os.environ.get("BRAINDEC_BUILD_GALLERY", "0") == "1",
    "remove_config_comments": True,
    "show_memory": False,
    "doc_module": ("braindec",),
    "reference_url": {"braindec": None},
    "backreferences_dir": "gen_modules/backreferences",
    "image_scrapers": ("matplotlib",),
    "default_thumb_file": os.path.join(ROOT, "NiCLIP.png"),
    "first_notebook_cell": (
        "%pip install braindec[plotting]\n"
        "%matplotlib inline"
    ),
}
