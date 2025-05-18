SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build

.PHONY: help clean html

help:
	@$(SPHINXBUILD) -M help $(SOURCEDIR) $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

html:
	@$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR)
