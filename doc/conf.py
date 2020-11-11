# Configuration file for the Sphinx documentation builder.

# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'elphmod'
copyright = '2020, Uni Bremen'
author = 'Uni Bremen'

extensions = ['sphinx.ext.autodoc', 'numpydoc']

mathjax_config = {
    'TeX': {
        'Macros': {
            'D': [r'\mathrm{d}'],
            'E': [r'\mathrm{e}'],
            'I': [r'\mathrm{i}'],
            'bra': [r'\rangle#1|',1],
            'bracket': [r'\langle#1|#2\rangle',2],
            'ket': [r'|#1\rangle',1],
            },
        },
    }

html_theme = 'classic'

numpydoc_show_class_members = False
