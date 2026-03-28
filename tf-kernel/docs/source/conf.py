# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../..'))

# -- Mock modules for autodoc on ReadTheDocs/CI (no CUDA) --------------------
autodoc_mock_imports = os.environ.get('SPHINX_AUTODOC_MOCK_MODULES', '').split(',')
autodoc_mock_imports = [m.strip() for m in autodoc_mock_imports if m.strip()]

# Always mock these CUDA-dependent modules for doc building
autodoc_mock_imports.extend([
    'torch',
    'tf_kernel',
])

# -- Project information -----------------------------------------------------
project = 'tf-kernel'
copyright = '2026, TeleFuser Team'
author = 'TeleFuser Team'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Todo settings
todo_include_todos = True

# -- Read the Docs specific configuration ------------------------------------
# Use Read the Docs theme locally as well
import sphinx_rtd_theme  # noqa: F401
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
