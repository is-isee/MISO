# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "Doxygen preprocessing..."
	@doxygen ../docs/doxygen/Doxyfile_cpu
	@doxygen ../docs/doxygen/Doxyfile_gpu
	@echo "=== Building CPU docs ==="
	@DOCS_TARGET=CPU $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/cpu" $(SPHINXOPTS)
	@echo "=== Building GPU docs ==="
	@DOCS_TARGET=GPU $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/gpu" $(SPHINXOPTS)
	@echo "=== Merging CPU and GPU docs ==="
	@DOCS_TARGET=MERGE $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "=== Cleaning up ==="
	@rm -rf "$(BUILDDIR)/cpu" "$(BUILDDIR)/gpu"

clean:
	rm -rf build source/api