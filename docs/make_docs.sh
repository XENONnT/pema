#!/usr/bin/env bash
make clean
rm -r source/reference
sphinx-apidoc -o source/reference ../pema
rm source/reference/modules.rst
make html
