import importlib.util
from pathlib import Path


APPS_ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "validate_readmes_for_test",
    APPS_ROOT / "scripts" / "validate_readmes.py",
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)
validate_readme = VALIDATOR.validate_readme


def write_readme(tmp_path: Path, *, install: str, run: str) -> Path:
    readme = tmp_path / "examples" / "classification" / "sample" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text(
        f"""# Sample

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | sample |
| Languages | C++, Python |
| Status | experimental |
| Binary Name | sample |
| Model | sample-model |

## Concept
Sample concept.

## Prerequisites
- A supported target.

## Install Apps
{install}

## Prepare the Model
Download the sample model.

## Run
{run}

## Source Files
- C++ source: `src/cpp/main.cpp`

## Development From Source
See [CONTRIBUTING.md](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
""",
        encoding="utf-8",
    )
    return readme


VALID_INSTALL = """Install the latest Neat Apps runtime.

```bash
sima-cli neat install apps
cd prebuilt-apps
```"""

VALID_RUN = """```bash
./examples/classification/sample/src/cpp/pre-built/sample
python3 examples/classification/sample/src/python/main.py
```"""


def test_installed_readme_contract_accepts_packaged_commands(tmp_path: Path) -> None:
    readme = write_readme(tmp_path, install=VALID_INSTALL, run=VALID_RUN)

    assert validate_readme(readme) == []


def test_installed_readme_contract_requires_latest_install(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL.replace("install apps", "install apps@v0.3.0"),
        run=VALID_RUN,
    )

    assert any(
        "sima-cli neat install apps" in error for error in validate_readme(readme)
    )


def test_installed_readme_contract_rejects_build_tree_binary(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN.replace(
            "./examples/classification/sample/src/cpp/pre-built/sample",
            "./build/examples/classification/sample/sample",
        ),
    )

    errors = validate_readme(readme)
    assert any("Source-only command './build/'" in error for error in errors)
    assert any("packaged C++ binary" in error for error in errors)


def test_installed_readme_contract_rejects_source_builds_in_install(
    tmp_path: Path,
) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL + "\n\n./build.sh --clean",
        run=VALID_RUN,
    )

    assert any(
        "Source-only command 'build.sh'" in error for error in validate_readme(readme)
    )
