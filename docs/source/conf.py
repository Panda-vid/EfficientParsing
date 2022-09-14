# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path.insert(0, (pathlib.Path(__file__).parents[2] / "efficient_parsing" / "src").resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Efficient Parsing'
copyright = '2022, David Schwenke'
author = 'David Schwenke'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'autoapi.extension',
    'sphinx.ext.autodoc.typehints'
]

templates_path = ['_templates']
exclude_patterns = []

# -- AutoAPI options ---------------------------------------------------------
autoapi_dirs = ['../../efficient_parsing/']
autodoc_typehints = 'description'
add_module_names = False
autoapi_python_use_implicit_namespaces = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
