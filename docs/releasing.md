# Releasing MERLIN (GitHub + PyPI)

This repository supports release automation through GitHub Actions.

## What is already configured

- Python package metadata and version live in `pyproject.toml` (`[project].version`).
- Release workflow lives in `.github/workflows/release.yml`.
- The workflow runs on Git tags like `v0.1.0`.
- It will:
  1. validate the tag version matches `pyproject.toml`,
  2. run tests,
  3. build wheel + sdist,
  4. create a GitHub Release with the artifacts,
  5. publish to PyPI.

## One-time repository setup

1. In GitHub, open **Settings → Secrets and variables → Actions**.
2. Add repository secret:
   - `PYPI_API_TOKEN` = your PyPI API token.
     Ensure this is configured for publishing to PyPI.
3. (Optional but recommended) create a protected `pypi` environment:
   - **Settings → Environments → New environment → `pypi`**
   - Add approval rules if desired.

## Release steps

1. Update package version in `pyproject.toml`.
   - Example: `0.1.0` → `0.1.1`
2. Commit and merge the version bump to your release branch (usually `main`).
3. Create and push a tag that matches that version with `v` prefix:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

4. Open **Actions** in GitHub and monitor the **Release** workflow.
5. After success:
   - GitHub Release is created with `.whl` and `.tar.gz` attached.
   - Package is published to PyPI.

## Versioning rules

- Use semantic versioning (`MAJOR.MINOR.PATCH`).
- Tag must be `v<version>` and match exactly:
  - tag `v0.1.1` ⇔ `version = "0.1.1"` in `pyproject.toml`.
- If they do not match, workflow fails intentionally.

## Manual dry run before tagging

From repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pip install build twine
python -m build
python -m twine check dist/*
```

If all commands pass, you are ready to tag and release.
