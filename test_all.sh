pre-commit run --show-diff-on-failure --color=always --all-files
poetry run pytest -W ignore::DeprecationWarning --verbose --cov=quantum_linear_systems tests
