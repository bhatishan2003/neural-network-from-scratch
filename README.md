## Table of Contents <!-- omit in toc -->

- [Installation](#installation)
- [Usage](#usage)
- [Development Notes](#development-notes)

## Installation

1. **Create a Virtual Environment [Optional, but recommended]**

    Run the following command to create a [virtual environment](https://docs.python.org/3/library/venv.html):

    ```bash
    python3 -m venv .venv
    ```

    - **Activate:**

        - **Windows (PowerShell):**

        ```bash
        .venv\Scripts\activate
        ```

        - **Linux/Mac (Bash):**

        ```bash
        source .venv/bin/activate
        ```

    - **Deactivate:**
        ```bash
        deactivate
        ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

-   Train the model

    Run training for a specified number of epochs (e.g., 100 epochs):

    ```bash
    python neural_network.py --num_epochs 100
    ```

-   Help

    Displays the available options and usage details:

    ```bash
    python neural_network.py --help
    ```

## Development Notes

-   Pre-commit

    We use pre-commit to automate linting of our codebase.

    -   Install hooks:
        ```bash
        pre-commit install
        ```
    -   Run Hooks manually (optional):
        ```bash
        pre-commit run --all-files
        ```

-   Ruff:

    -   Lint and format:
        ```bash
        ruff check --fix
        ruff format
        ```
