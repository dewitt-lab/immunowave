import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "immunowave"
copyright = "2025, immunowave team"
author = "immunowave team"
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",  # Add nbsphinx for Jupyter notebook support
]

# Configure nbsphinx for Jupyter notebooks
nbsphinx_execute = "never"  # Don't execute notebooks when building docs
nbsphinx_allow_errors = True  # Continue building docs if notebook cells have errors

# Configure MyST-Parser to enable special syntax for notebooks
myst_enable_extensions = [
    "dollarmath",  # For LaTeX math with $
    "colon_fence",  # For code fences using :::
    "smartquotes",  # Smart quotes
    "replacements",  # Common text replacements
    "linkify",  # Auto-detect URLs
    "substitution",  # Allow variables like {{version}}
]

# Enable cross-references to Python objects
myst_heading_anchors = 3  # Generate anchors for h1-h3

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Fix for _static directory warning ---------------------------------------
# Create _static directory if it doesn't exist
import os

static_dir = os.path.join(os.path.dirname(__file__), "_static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# -- Intersphinx mapping ----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "equinox": ("https://docs.kidger.site/equinox/", None),
    "diffrax": ("https://docs.kidger.site/diffrax/", None),
}
