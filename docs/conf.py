# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add the parent directory (project root) to sys.path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information

project = 'Ethics in AI'
copyright = '2025, Lyudmil'
author = 'Lyudmil'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    # 'myst_parser',
    'myst_nb',
    'sphinx.ext.viewcode',
    'nbsphinx',  # Enable nbsphinx
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'