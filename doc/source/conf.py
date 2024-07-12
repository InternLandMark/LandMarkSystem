# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "LandmarkSystem"
copyright = "2024, landmark team"  # pylint: disable=W0622
author = "landmark team"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# GitHub integration
html_context = {
    "display_github": True,
    "github_user": "InternLandMark",
    "github_repo": "LandMarkSystem",
    "github_version": "main",
    "conf_py_path": "/doc/source/",
}

# sys.path.insert(0, "/home/doc/checkouts/readthedocs.org/user_builds/landmarksystem/checkouts/latest/")
# sys.path.insert(0, "/home/doc/checkouts/readthedocs.org/user_builds/landmarksystem/checkouts/latest/landmark")
# print(f"{sys.path=}")

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
