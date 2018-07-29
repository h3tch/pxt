# pxt
A small python package to easily compile code of other languages and link them to python code in a simple and flexible manner.

# Running tests

Run ``python setup.py test``

# Building the documentation

In the doc folder call::

```
sphinx-quickstart doc --quiet --sep --dot=. --project=pxt --author=h3tch --ext-autodoc --extensions=sphinx.ext.napoleon --makefile --batchfile

sphinx-apidoc -f -o `source/ ../pxt/

make html
```

