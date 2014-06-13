# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

flake:
	@if command -v flake8 > /dev/null; then \
           echo "Running flake8"; \
		flake8 --count expyfun examples; \
           echo "Done."; \
	else \
           echo "flake8 not found, please install it!"; \
     fi;


in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

nosetests:
	rm -f .coverage
	$(NOSETESTS) expyfun

test: clean nosetests flake

test-doc:
	$(NOSETESTS) --with-doctest --doctest-tests --doctest-extension=rst doc/ doc/source/
