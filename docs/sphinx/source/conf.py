# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
## -- imports

from importlib.metadata import version as vn

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torchspin'
copyright = '2026, Alec Follmer'
author = 'Alec Follmer'
release = '.'.join(vn('torchspin').split('.'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

extensions = [
	'sphinx.ext.napoleon', # for numpy in autodoc
	'sphinx.ext.autodoc',  # for automatic documentation of docstrings
	'sphinx.ext.mathjax',  # for latex support in html 
	'sphinx_copybutton',   # allow code copy button in examples
	'sphinx_design',	   # enable design elements like grids
]
language = "en"
templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_show_sourcelink = False 
#html_favicon = "_static/favi_redinv.svg"
#html_logo = "_static/logo.svg"  # can use if you want logos in docs

html_context = {
	"default_mode": "dark",
	"github_user": "follmerlab",
	"github_repo" : "torchspin",
	"github_version" : "main",
	"doc_path":"doc",
}
