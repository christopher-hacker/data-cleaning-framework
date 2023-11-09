"""Scaffolds a new project."""

from importlib.metadata import version
import os
from typing import Literal
import jinja2 as j2
from pydantic import BaseModel, EmailStr
import questionary

DCF_VERSION = version("data_cleaning_framework")


class ProjectConfig(BaseModel):
    """Model for project config options."""

    project_name: str = "data-cleaning-project"
    description: str = ""
    version: str = "0.1.0"
    author: str = "Your Name"
    email: EmailStr = "yourname@email.com"
    src_directory: str = "src"
    dependency_manager: Literal["poetry", "pip", "conda"] = "poetry"
    dcf_version: str = DCF_VERSION


def create_with_warning(path: str, content: str) -> None:
    """Creates a file, and warns if it already exists."""
    if os.path.exists(path):
        questionary.print(
            f"Warning: {path} already exists, skipping creation",
            style="bold fg:ansiyellow",
        )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def create_pip_project(project_dir, environment, config):
    """Creates a pip project and creates a virtual environment"""
    create_with_warning(
        os.path.join(project_dir, "setup.py"),
        environment.get_template("setup.py.j2").render(config=config),
    )
    create_with_warning(
        os.path.join(project_dir, "requirements.txt"),
        environment.get_template("requirements.txt.j2").render(config=config),
    )


def create_poetry_project(project_dir, environment, config):
    """Creates a poetry project"""
    create_with_warning(
        os.path.join(project_dir, "pyproject.toml"),
        environment.get_template("pyproject.toml.j2").render(config=config),
    )
    create_with_warning(
        os.path.join(project_dir, "poetry.toml"),
        environment.get_template("poetry.toml.j2").render(config=config),
    )


def main():
    """Scaffolds a new project."""
    if not questionary.confirm(
        "This will create a new project in the current directory. Continue?"
    ).ask():
        return

    properties = ProjectConfig.model_json_schema()["properties"]
    config = ProjectConfig(
        project_name=questionary.text(
            "Project name:", default=properties["project_name"]["default"]
        ).ask(),
        description=questionary.text(
            "Project description:", default=properties["description"]["default"]
        ).ask(),
        version=questionary.text(
            "Project version:", default=properties["version"]["default"]
        ).ask(),
        author=questionary.text(
            "Author:", default=properties["author"]["default"]
        ).ask(),
        email=questionary.text("Email:", default=properties["email"]["default"]).ask(),
        src_directory=questionary.text(
            "Source directory:", default=properties["src_directory"]["default"]
        ).ask(),
        dependency_manager=questionary.select(
            "Dependency manager:",
            choices=properties["dependency_manager"]["enum"],
            default=properties["dependency_manager"]["default"],
        ).ask(),
    )

    environment = j2.Environment(
        loader=j2.PackageLoader("data_cleaning_framework", "templates")
    )

    # first create the src directory
    questionary.print("Creating src files", style="bold fg:ansiblue")
    project_dir = os.getcwd()
    src_dir = os.path.join(project_dir, config.src_directory)
    os.makedirs(src_dir, exist_ok=True)

    # create the python files
    # __init__.py
    create_with_warning(
        os.path.join(src_dir, "__init__.py"), "# Path: src/__init__.py\n"
    )
    # schema.py
    create_with_warning(
        os.path.join(src_dir, "schema.py"),
        environment.get_template("schema.py.j2").render(config=config),
    )
    # cleaners.py
    create_with_warning(
        os.path.join(src_dir, "cleaners.py"),
        environment.get_template("cleaners.py.j2").render(config=config),
    )

    # create project files for each dependency manager and create a virtual environment
    match config.dependency_manager:
        case "poetry":
            questionary.print(
                "Creating environment files and virtual environment using poetry",
                style="bold fg:ansiblue",
            )
            create_poetry_project(project_dir, environment, config)
            os.system("poetry install && poetry shell")

        case "pip":
            questionary.print(
                "Creating virtual environment in project",
                style="bold fg:ansiblue",
            )
            create_pip_project(project_dir, environment, config)
            if not os.path.exists(".venv"):
                os.system("python -m venv .venv")
            os.system("source .venv/bin/activate && pip install -r requirements.txt")

        case "conda":
            raise NotImplementedError("Conda not yet implemented")

        case _:
            raise ValueError(
                f"Dependency manager {config.dependency_manager} not recognized"
            )
