# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Introduction to CUDA Programming"
copyright = "2024, National Computational Infrastructure"
author = "Joseph John"

release = "0.1"
version = "0.1.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "path_to_docs": "docs/source",
    "repository_url": "https://github.com/NCI900-Training-Organisation/intro-to-cuda.git",
    "use_repository_button": True,
    "home_page_in_toc": True,
    "back_to_top_button": True,
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "icon_links": [
        {
            "name": "NCI Documentation",
            "url": "https://opus.nci.org.au/spaces/Help/pages/12583138/NCI+Help",
            "icon": "fa-brands fa-confluence",
            "type": "fontawesome",
        },
        {
            "name": "Courses",
            "url": "https://nci900-training-organisation.github.io/learning-resources/courses.html",
            "icon": "fa-solid fa-graduation-cap",
            "type": "fontawesome",
        },
    ],
}

# -- Options for EPUB output
epub_show_urls = "footnote"
