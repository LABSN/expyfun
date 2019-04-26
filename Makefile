# simple makefile to simplify repetetive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTEST ?= pytest
CTAGS ?= ctags

CODESPELL_SKIPS ?= "*.log,*.doctree,*.pickle,*.png,*.js,*.html,*.orig"
CODESPELL_DIRS ?= expyfun/ doc/ examples/

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

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count expyfun examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;


in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

pytest:
	rm -f .coverage
	$(PYTEST) expyfun

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

test: clean pytest flake codespell-error

test-doc:
	$(PYTEST) --doctest-modules --doctest-ignore-import-errors expyfun

version:
	@expr substr `git rev-parse HEAD` 1 7

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle expyfun

docstyle: pydocstyle

check-manifest:
	check-manifest --ignore .DS_Store