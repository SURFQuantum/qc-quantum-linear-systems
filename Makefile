PYTHON3         = python3
TEST_DIR        = tests
SOURCE_DIR     	= quantum_linear_systems
REQUIREMENTS    = requirements.txt
MINCOV       	= 90

all: install test

install:
	@echo -e "\n*** Installing $(SOURCE_DIR) package locally"
	@$(PYTHON3) -m pip install -e .

lint:
	@echo -e "\n*** Checking that $(SOURCE_DIR) conforms to PEP8 coding style"
	@$(PYTHON3) -m flake8 ${SOURCE_DIR} ${TEST_DIR}

test:
	@echo -e "\n*** Running unit tests for $(SOURCE_DIR)"
	@$(PYTHON3) -m pytest --verbose $(TEST_DIR) -p no:warnings

test-cov:
	@echo -e "\n*** Running unit tests for $(SOURCE_DIR) including coverage"
	@$(PYTHON3) -m pytest --verbose --cov=${SOURCE_DIR} $(TEST_DIR)

clean:
	@find . -name "*.pyc" -delete

verify: clean install lint test-cov _verified

_verified:
	@echo "$(SOURCE_DIR) is verified :)"
