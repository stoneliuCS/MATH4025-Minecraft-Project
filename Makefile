SHELL := /bin/bash

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(PROJECT_ROOT)/.venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate

# macOS Java 8 setup per README
JAVA_HOME_8 := $(shell /usr/libexec/java_home -v 1.8 2>/dev/null)

# Prefer Python 3.10 or 3.9 per README
PYTHON_BIN := $(shell command -v python3.10 || command -v python3.9 || command -v python3)
PYTHON_VERSION := $(shell $(PYTHON_BIN) -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)

PYTHONPATH := $(PROJECT_ROOT)
export PYTHONPATH

INTERACTIVE_PORT ?= 6666
MINERL_SRC ?= $(PROJECT_ROOT)/minerl

.PHONY: help env venv check-java check-python print-env run interactor patch-minerl

help:
	@echo "Targets:"
	@echo "  make venv        Create .venv with Python 3.9/3.10"
	@echo "  make env         Verify Java 8, Python 3.9/3.10, and .venv"
	@echo "  make print-env   Print exports for current shell"
	@echo "  make run         Run the main Python entrypoint"
	@echo "  make interactor  Run MineRL interactor on port $(INTERACTIVE_PORT)"
	@echo "  make patch-minerl  Patch/rebuild MCP-Reborn and copy into venv"

venv:
	@if [ -z "$(PYTHON_BIN)" ]; then \
		echo "No python3 found. Install Python 3.9 or 3.10 first."; \
		exit 1; \
	fi
	@$(PYTHON_BIN) -m venv "$(VENV_DIR)"
	@echo "Created $(VENV_DIR)"

check-java:
	@if [ -z "$(JAVA_HOME_8)" ]; then \
		echo "Java 8 not found. Install temurin@8 (brew) and try again."; \
		exit 1; \
	fi
	@echo "Using JAVA_HOME=$(JAVA_HOME_8)"
	@JAVA_HOME="$(JAVA_HOME_8)" PATH="$(JAVA_HOME_8)/bin:$$PATH" java -version

check-python:
	@if [ -z "$(PYTHON_BIN)" ]; then \
		echo "No python3 found. Install Python 3.9 or 3.10 first."; \
		exit 1; \
	fi
	@if [ "$(PYTHON_VERSION)" != "3.10" ] && [ "$(PYTHON_VERSION)" != "3.9" ]; then \
		echo "Python $(PYTHON_VERSION) found. Please use Python 3.9 or 3.10."; \
		exit 1; \
	fi
	@echo "Using Python $(PYTHON_VERSION) at $(PYTHON_BIN)"

env: check-java check-python
	@if [ ! -f "$(VENV_ACTIVATE)" ]; then \
		echo "Missing $(VENV_ACTIVATE). Run 'make venv' first."; \
		exit 1; \
	fi
	@echo "Venv ok: $(VENV_DIR)"
	@echo "PYTHONPATH=$(PYTHONPATH)"
	@echo ""
	@echo "Run this in your shell to activate the env with Java 8:"
	@echo "  export JAVA_HOME=\"$(JAVA_HOME_8)\""
	@echo "  export PATH=\"$$JAVA_HOME/bin:$$PATH\""
	@echo "  source \"$(VENV_ACTIVATE)\""
	@echo "  export PYTHONPATH=\"$(PYTHONPATH)\""

print-env:
	@echo "export JAVA_HOME=\"$(JAVA_HOME_8)\""
	@echo "export PATH=\"$$JAVA_HOME/bin:$$PATH\""
	@echo "source \"$(VENV_ACTIVATE)\""
	@echo "export PYTHONPATH=\"$(PYTHONPATH)\""

run: env
	@JAVA_HOME="$(JAVA_HOME_8)" PATH="$(JAVA_HOME_8)/bin:$$PATH" \
	"$(VENV_DIR)/bin/python" "$(PROJECT_ROOT)/model/main.py"

interactor: env
	@JAVA_HOME="$(JAVA_HOME_8)" PATH="$(JAVA_HOME_8)/bin:$$PATH" \
	"$(VENV_DIR)/bin/python" -m minerl.interactor $(INTERACTIVE_PORT)
