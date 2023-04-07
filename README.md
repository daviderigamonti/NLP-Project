# NLP_Project
Repository dedicated to the project held for the Natural Language Processing course at Politecnico di Milano a.a. 2022-2023 with Professor Carman. 

## Notes

### Poetry
Poetry enables proper dependency management in Python, a detailed documentation containing instructions regarding the setup can be found [here](https://python-poetry.org/docs/basic-usage).

The initial virtual environment can be setup using:
```
poetry install
```
The jupyter notebook server can be run inside the environment with the command:
```
poetry run jupyter notebook
```
In the notebooks and in our code, it's possible to reference code from our library through the `nlp_project` module.

### Pre-commit
To ensure that formatting and git quality remains high it is recommended to add [`pre-commit`](https://pre-commit.com) to the git hooks, this can be done by running the following command after the `poetry` installation:
```
poetry run pre-commit install
```

If needed, pre-commit can be bypassed by using the `--no-verify` flag on the `git commit` command.
