# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "Doxygen preprocessing..."
	@doxygen ../docs/doxygen/Doxyfile
	@echo "=== Building C++ docs ==="
	@DOCS_TARGET=CPP $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)/cpu" $(SPHINXOPTS) $(O)
	@echo "=== Merging CPU and GPU docs ==="
	@DOCS_TARGET=MERGE $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "=== Cleaning up ==="
	@rm -rf "$(BUILDDIR)/cpu" "$(BUILDDIR)/gpu"

clean:
	rm -rf _build source/api doxygen/_build
