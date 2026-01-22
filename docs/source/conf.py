# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- import modules
import pymiso
# import os

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
    "sphinx_automodapi.automodapi",  # automatic API documentation
    "sphinx_multiversion",  # multiple version support
    "sphinx_copybutton",  # copy button for code blocks
    "breathe",  # Doxygen integration
    "sphinxcontrib.bibtex",  # bibliography management
    "myst_parser",  # Markdown support
    "sphinx.ext.mathjax",  # MathJax for math rendering
]

mathjax3_config = {
    "tex": {
        "macros": {
            "bm": ["{\\boldsymbol{#1}}", 1],
        }
    }
}

myst_enable_extensions = [
    "dollarmath",  # $...$ や $$...$$ の数式を使えるようにする
    "amsmath",  # align環境などを有効化
]

autodoc_default_options = {
    "special-members": "__init__",
}

breathe_projects = {"CPP": "../doxygen/_build/xml"}
breathe_default_project = "CPP"

extensions.append("exhale")
exhale_args = {
    "containmentFolder": "./api/cpp",
    "rootFileName": "root.rst",
    "rootFileTitle": "MISO C++ API Reference",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
}

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# pip install sphinx-rtd-theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
#

bibtex_bibfiles = ["reference.bib"]
bibtex_reference_style = "author_year"
