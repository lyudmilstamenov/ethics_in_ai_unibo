# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Add your project root to sys.path


# -- Project information -----------------------------------------------------

project = 'Ethics in AI Project'
author = 'Lyudmil Stamenov'
release = '0.1.0'  # Version number


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',      # Support for Google-style or NumPy-style docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.todo',          # Support for TODO directives
    'myst_parser',              # Markdown support (optional)
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
}

# -- Napoleon settings (if using Google or NumPy docstrings) -----------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for TODOs -------------------------------------------------------

todo_include_todos = True
