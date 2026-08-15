"""Compile the optional Kubeflow training pipeline to KFP v2 IR YAML."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Sequence


DEFAULT_OUTPUT_PATH = Path("pipelines/startup_training_pipeline.yaml")


def _missing_kfp_error(error: ModuleNotFoundError) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "Kubeflow compilation requires the optional KFP SDK. Install "
        "requirements-kubeflow.txt and retry."
    )


def compile_startup_pipeline(
    component_image: str,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Compile the pipeline without connecting to a Kubeflow cluster."""

    try:
        from kfp import compiler

        from .components import validate_component_image
        from .pipeline import create_startup_training_pipeline
    except ModuleNotFoundError as error:
        if error.name == "kfp" or (error.name or "").startswith("kfp."):
            raise _missing_kfp_error(error) from error
        raise

    image = validate_component_image(component_image)
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Kubeflow pipeline output must end with '.yaml' or '.yml'.")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Compiled pipeline already exists: {destination}. "
            "Use --overwrite to replace it."
        )
    if destination.exists() and not destination.is_file():
        raise ValueError(f"Pipeline output is not a file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        temporary_path.unlink()

        pipeline = create_startup_training_pipeline(image)
        compiler.Compiler().compile(
            pipeline_func=pipeline,
            package_path=str(temporary_path),
        )
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without importing the KFP SDK."""

    parser = argparse.ArgumentParser(
        description="Compile the startup training pipeline to KFP v2 IR YAML."
    )
    parser.add_argument(
        "--component-image",
        required=True,
        help=(
            "Registry image used by every runtime component; an immutable "
            "digest is recommended"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination .yaml file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing compiled pipeline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    """Compile from command-line arguments and return the created path."""

    arguments = build_parser().parse_args(argv)
    output = compile_startup_pipeline(
        component_image=arguments.component_image,
        output_path=arguments.output,
        overwrite=arguments.overwrite,
    )
    print(f"Compiled Kubeflow pipeline: {output}")
    return output


if __name__ == "__main__":
    main()
