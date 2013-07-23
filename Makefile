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

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test:
	$(NOSETESTS) expyfun

test-doc:
	$(NOSETESTS) --with-doctest --doctest-tests --doctest-extension=rst doc/ doc/source/

codespell:
	# The *.fif had to be there twice to be properly ignored (!)
	codespell.py -w -i 3 -S="*.fif,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.coverage,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii" ./dictionary.txt -r .

