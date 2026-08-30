# Pre-commit Hooks Guide

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality and consistency.

## What is Pre-commit?

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. When you commit code, pre-commit automatically runs configured checks and fixes to ensure code quality.

## Installation

### Quick Setup

```bash
# Install development dependencies (includes pre-commit)
uv pip install -e ".[dev]"

# Install the pre-commit hooks
pre-commit install
```

### Manual Setup

```bash
# Install pre-commit separately
uv pip install pre-commit

# Install hooks
pre-commit install
```

## Configured Hooks

Our pre-commit configuration (`.pre-commit-config.yaml`) includes:

### 1. General File Checks
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline
- **check-yaml**: Validates YAML files
- **check-toml**: Validates TOML files
- **check-json**: Validates JSON files
- **check-added-large-files**: Prevents large files (>1MB)
- **check-merge-conflict**: Detects merge conflict markers
- **mixed-line-ending**: Enforces LF line endings
- **detect-private-key**: Prevents committing secrets

### 2. Python Code Quality (Ruff)
- **ruff**: Fast Python linter (replaces flake8, isort, etc.)
  - Automatically fixes issues when possible
  - Checks code style, imports, complexity
  - Enforces PEP 8 guidelines

- **ruff-format**: Fast Python formatter (replaces black)
  - Auto-formats code to consistent style
  - 88-character line length
  - Double quotes for strings

### 3. Type Checking (mypy)
- **mypy**: Static type checker
  - Validates type hints
  - Catches type-related bugs
  - Ignores missing imports (configured in pyproject.toml)

### 4. Security (Bandit)
- **bandit**: Security vulnerability scanner
  - Detects common security issues
  - Configured to skip test assertions (B101)
  - Excludes test directories

### 5. Documentation (pydocstyle)
- **pydocstyle**: Docstring style checker
  - Enforces Google-style docstrings
  - Skips test files
  - Ensures documentation quality

### 6. Markdown Linting
- **markdownlint**: Markdown style checker
  - Fixes formatting issues
  - Enforces consistent style
  - Configured in `.markdownlint.yaml`

### 7. YAML Formatting
- **pretty-format-yaml**: YAML formatter
  - Auto-formats YAML files
  - 2-space indentation
  - Consistent formatting

## Usage

### Automatic (Recommended)

Pre-commit hooks run automatically when you commit:

```bash
git add .
git commit -m "Your commit message"
# Hooks run automatically
```

If hooks fail or make changes:
```bash
# Review the changes
git diff

# Stage the auto-fixed files
git add -u

# Commit again
git commit -m "Your commit message"
```

### Manual

Run hooks manually without committing:

```bash
# Run on staged files
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

## Common Scenarios

### First-time Setup

```bash
# Install and run on all files
pre-commit install
pre-commit run --all-files
```

This will format all existing files and show any issues.

### Before Committing

```bash
# Check what will run
pre-commit run --all-files

# Fix issues manually if needed
ruff check . --fix
ruff format .

# Commit
git commit -m "Your message"
```

### Skipping Hooks (Use Sparingly!)

```bash
# Skip all hooks
git commit --no-verify -m "Emergency fix"

# Skip specific hooks
SKIP=mypy git commit -m "Skip type checking"
SKIP=mypy,bandit git commit -m "Skip multiple hooks"
```

⚠️ **Warning:** Only skip hooks when absolutely necessary (e.g., emergency fixes).

### Updating Hooks

```bash
# Update to latest hook versions
pre-commit autoupdate

# Clean and reinstall
pre-commit clean
pre-commit install
```

## Configuration

### Pre-commit Config (`.pre-commit-config.yaml`)

Main configuration file. Modify to:
- Add/remove hooks
- Change hook versions
- Adjust hook arguments

Example:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

### Ruff Config (`.ruff.toml`)

Ruff-specific configuration:
- Line length (88)
- Enabled rules
- Ignored rules
- Per-file ignores

### Markdownlint Config (`.markdownlint.yaml`)

Markdown linting rules:
- Line length (120)
- Allowed HTML elements
- Heading styles

### Python Config (`pyproject.toml`)

Python tool configurations:
- mypy settings
- bandit settings
- pytest settings

## Troubleshooting

### Hook Fails with "command not found"

```bash
# Reinstall pre-commit
uv pip install --upgrade pre-commit
pre-commit clean
pre-commit install
```

### Hooks are Slow

```bash
# Skip slow hooks during development
SKIP=mypy git commit -m "Your message"

# Run manually before final commit
pre-commit run --all-files
```

### Auto-fixes Create Conflicts

```bash
# Review changes
git diff

# If acceptable, stage and commit
git add -u
git commit -m "Your message"

# If not, revert and fix manually
git checkout -- .
# Make manual fixes
git add .
git commit -m "Your message"
```

### Mypy Type Errors

```bash
# Run mypy manually to see details
mypy vllm_bart_plugin/

# Skip mypy for this commit
SKIP=mypy git commit -m "WIP: fix types later"
```

### Ruff Formatting Conflicts

```bash
# Run ruff manually
ruff format .
ruff check . --fix

# Stage and commit
git add .
git commit -m "Your message"
```

## Best Practices

### 1. Run Before Committing
```bash
# Check before committing
pre-commit run --all-files
```

### 2. Fix Issues Early
```bash
# Format code while developing
ruff format .
ruff check . --fix
```

### 3. Don't Skip Without Reason
- Only use `--no-verify` for emergencies
- Fix issues instead of skipping
- Document why if you must skip

### 4. Keep Hooks Updated
```bash
# Update monthly
pre-commit autoupdate
```

### 5. Review Auto-fixes
```bash
# Always review what pre-commit changed
git diff
```

## Integration with Development

### IDE Integration

Most IDEs can run pre-commit on save:

**VSCode** (`.vscode/settings.json`):
```json
{
  "emeraldwalk.runonsave": {
    "commands": [
      {
        "match": "\\.py$",
        "cmd": "ruff format ${file} && ruff check ${file} --fix"
      }
    ]
  }
}
```

**PyCharm**:
- Settings → Tools → File Watchers
- Add ruff and mypy watchers

### CI/CD Integration

Pre-commit can run in CI:

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit
on: [push, pull_request]
jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - uses: pre-commit/action@v3.0.0
```

## Performance Tips

### 1. Use Ruff Instead of Multiple Tools
Ruff replaces:
- black (formatting)
- isort (import sorting)
- flake8 (linting)
- pyupgrade (syntax upgrades)

It's much faster than running all these separately!

### 2. Skip Heavy Hooks During Development
```bash
# Create alias in .bashrc or .zshrc
alias gcm='SKIP=mypy git commit -m'

# Use for quick commits
gcm "WIP: testing"

# Run full check before PR
pre-commit run --all-files
```

### 3. Cache Hook Environments
Pre-commit caches hook environments. Don't clean unless necessary:
```bash
# Only if hooks are broken
pre-commit clean
pre-commit install
```

## Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Pydocstyle Documentation](http://www.pydocstyle.org/)
