# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- import modules
import pyMISO

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MISO"
copyright = "2025, Integrated Studies, ISEE, Nagoya University"
author = "Hideyuki Hotta, Haruhisa Iijima"
release = "0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # automatic documentation from docstrings
    "sphinx.ext.napoleon",  # support for Google style docstrings
    "sphinx.ext.viewcode",  # ソースコードのリンク
    "sphinx.ext.intersphinx",  # 他のドキュメントへのリンク
    "sphinx.ext.todo",  # todo directives
    "sphinx.ext.imgmath",  # math rendering
    "sphinx_automodapi.automodapi",  # automatic API documentation
    "sphinx_multiversion",  # multiple version support
    "breathe",  # Doxygen integration
]

# pip install  sphinx-automodapi sphinx-multiversion breathe exhale

breathe_projects = {
    "MISO-CPU": "../doxygen/cpu/xml",
    "MISO-GPU": "../doxygen/gpu/xml",
}

import os

target = os.environ.get("DOCS_TARGET", "CPU")

if target == "CPU":
    extensions.append("exhale")
    breathe_default_project = "MISO-CPU"
    exhale_args = {
        "containmentFolder": "./api/cpp_cpu",
        "rootFileName": "root.rst",
        "rootFileTitle": "MISO CPU API Reference",
        "doxygenStripFromPath": "..",
        "createTreeView": True,
    }
elif target == "GPU":
    extensions.append("exhale")
    breathe_default_project = "MISO-GPU"
    exhale_args = {
        "containmentFolder": "./api/cpp_gpu",
        "rootFileName": "root.rst",
        "rootFileTitle": "MISO GPU API Reference",
        "doxygenStripFromPath": "..",
        "createTreeView": True,
    }
else:
    breathe_default_project = "MISO-CPU"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pip install sphinx-rtd-theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
#
