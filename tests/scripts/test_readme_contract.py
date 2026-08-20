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


def write_readme(
    tmp_path: Path,
    *,
    install: str,
    run: str,
    concept: str = "Classifies one image with a compiled model and prints the result.",
    configure: str | None = (
        "Open `${APP_DIR}/src/common/config.yaml` and set `model.path`."
    ),
    preview: str = "![Sample preview](../../../portal/assets/examples/classification/sample/image.png)",
    status: str = "stable",
) -> Path:
    readme = tmp_path / "examples" / "classification" / "sample" / "README.md"
    readme.parent.mkdir(parents=True)
    configure_section = (
        f"## Configure\n{configure}\n\n" if configure is not None else ""
    )
    readme.write_text(
        f"""# Sample

## Metadata
| Field | Value |
| --- | --- |
| Category | classification |
| Difficulty | Beginner |
| Tags | sample |
| Languages | C++, Python |
| Status | {status} |
| Binary Name | sample |
| Model | sample-model |

## Concept
{concept}

## Preview
{preview}

## Prerequisites
- A supported target.

## Install Apps
{install}

## Prepare the Model
Download the sample model.

{configure_section}## Run
{run}

## Source Files
- C++ source: `src/cpp/main.cpp`

## Development From Source
See [CONTRIBUTING.md](https://github.com/sima-neat/apps/blob/main/CONTRIBUTING.md).
""",
        encoding="utf-8",
    )
    preview = (
        tmp_path
        / "portal"
        / "assets"
        / "examples"
        / "classification"
        / "sample"
        / "image.png"
    )
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"png")
    return readme


VALID_INSTALL = """Install the latest Neat Apps runtime.

```bash
sima-cli neat install apps
cd prebuilt-apps
APP_DIR=examples/classification/sample
```"""

VALID_STANDALONE_INSTALL = """Fetch only this example.

```bash
curl -fsSL https://raw.githubusercontent.com/sima-neat/apps/main/scripts/get-example.sh | bash -s -- sample
cd sample
```"""

VALID_RUN = """```bash
./${APP_DIR}/src/cpp/pre-built/sample
python3 ${APP_DIR}/src/python/main.py
```"""

VALID_STANDALONE_RUN = """```bash
./src/cpp/pre-built/sample
python3 src/python/main.py
```"""


def test_installed_readme_contract_accepts_packaged_commands(tmp_path: Path) -> None:
    readme = write_readme(tmp_path, install=VALID_INSTALL, run=VALID_RUN)

    assert validate_readme(readme) == []


def test_readme_contract_accepts_single_example_install(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_STANDALONE_INSTALL,
        run=VALID_STANDALONE_RUN,
    )

    assert validate_readme(readme) == []


def test_readme_contract_rejects_experimental_status(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        status="experimental",
    )

    assert "Invalid Status 'experimental'. Must be one of: stable" in validate_readme(
        readme
    )


def test_readme_contract_requires_configure(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        configure=None,
    )

    assert "Missing required section: ## Configure" in validate_readme(readme)


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
            "./${APP_DIR}/src/cpp/pre-built/sample",
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


def test_readme_contract_rejects_filler_summary(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        concept="This example classifies one image with a sample model.",
    )

    assert any("must start with what" in error for error in validate_readme(readme))


def test_readme_contract_rejects_long_summary(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        concept="Classifies images with a sample model and writes results. " * 5,
    )

    assert any("at most 200 characters" in error for error in validate_readme(readme))


def test_readme_contract_rejects_missing_preview_asset(tmp_path: Path) -> None:
    readme = write_readme(tmp_path, install=VALID_INSTALL, run=VALID_RUN)
    preview = (
        tmp_path
        / "portal"
        / "assets"
        / "examples"
        / "classification"
        / "sample"
        / "image.png"
    )
    preview.unlink()

    assert any("Preview image does not exist" in error for error in validate_readme(readme))


def test_readme_contract_rejects_fenced_preview(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        preview="""```md
![Sample preview](../../../portal/assets/examples/classification/sample/image.png)
```""",
    )

    assert any("Preview must contain a Markdown image" in error for error in validate_readme(readme))


def test_readme_contract_rejects_commented_preview(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        preview="""<!--
![Sample preview](../../../portal/assets/examples/classification/sample/image.png)
-->""",
    )

    assert any("Preview must contain a Markdown image" in error for error in validate_readme(readme))


def test_readme_contract_rejects_inline_code_preview(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        preview="`![Sample preview](../../../portal/assets/examples/classification/sample/image.png)`",
    )

    assert any("Preview must contain a Markdown image" in error for error in validate_readme(readme))


def test_readme_contract_rejects_raw_html_preview(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        preview="""<pre>
![Sample preview](../../../portal/assets/examples/classification/sample/image.png)
</pre>""",
    )

    assert "Preview must not contain raw HTML" in validate_readme(readme)


def test_readme_contract_rejects_repeated_bundle_path(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN
        + "\npython3 examples/classification/sample/src/python/main.py",
    )

    assert any("reuse APP_DIR" in error for error in validate_readme(readme))


def test_readme_contract_allows_focused_config_override(tmp_path: Path) -> None:
    readme = write_readme(
        tmp_path,
        install=VALID_INSTALL,
        run=VALID_RUN,
        configure="""Create a short-run override:

```yaml
runtime:
  frames: 30
```""",
    )

    assert validate_readme(readme) == []
