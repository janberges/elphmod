# Configuration file for the Sphinx documentation builder.

# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'elphmod'
copyright = '2017-2024 elphmod Developers'
author = 'Uni Bremen'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'numpydoc',
    'm2r2',
    ]

exclude_patterns = ['README.md']

mathjax_path = ('https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js'
    '?config=TeX-AMS-MML_HTMLorMML')

mathjax_config = {
    'TeX': {
        'Macros': {
            'D': [r'\mathrm{d}'],
            'E': [r'\mathrm{e}'],
            'I': [r'\mathrm{i}'],
            'bra': [r'\langle#1|', 1],
            'bracket': [r'\langle#1|#2\rangle', 2],
            'ket': [r'|#1\rangle', 1],
            },
        },
    }

rst_epilog = '''
.. include:: <isogrk1.txt>
'''

html_theme = 'sphinx_rtd_theme'
html_logo = '../logo/elphmod.svg'
html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': '#e7f2fa',
    }

numpydoc_show_class_members = False
