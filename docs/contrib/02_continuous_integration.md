# Continuous Integration in pySDC

Any commit in `pySDC` are tested by GitHub continuous integration (CI). You can see in in the [action panel](https://github.com/Parallel-in-Time/pySDC/actions) the tests for each branches.
Those tests are currently divided in three main categories : [code linting](#code-linting), [code testing](#code-testing) and [code coverage](#code-coverage).
Finally, the CI also build artifacts that are used to generate the documentation website (see http://parallel-in-time.org/pySDC/), more details given in the [documentation generation](#documentation-generation) section.

## Code linting

Code style linting is performed using [black](https://black.readthedocs.io/en/stable/) and [flakeheaven](https://flakeheaven.readthedocs.io/en/latest/) for code syntax checking. In particular, `black` is used to check compliance with (most of) [PEP-8 guidelines](https://peps.python.org/pep-0008/).

Those tests are conducted for each commit (even for forks), but you can also run it locally in the root folder of `pySDC` before pushing any commit :

```bash
# Install required packages (works also with conda/mamba)
pip install black flakeheaven flake8-comprehensions flake8-bugbear
# First : test code style linting with black
black pySDC --check --diff --color
# Second : test code syntax with flakeheaven
flakeheaven lint --benchmark pySDC
```

> :bell: To avoid any error about formatting (`black`), you can simply use this program to reformat directly your code using the command :
>
> ```bash
> black pySDC
> ```

Some style rules that are automatically enforced :

- lines should be not longer than 120 characters
- arithmetic operators (`+`, `*`, ...) should be separated with variables by one empty space

## Code testing

This is done using [pytest](https://docs.pytest.org/en/7.2.x/), and runs all the tests written in the `pySDC/tests` folder. You can run those locally in the root folder of `pySDC` using :

```bash
# Install required packages (works also with conda/mamba)
pip install pytest<7.2.0 pytest-benchmark coverage[toml]
# Run tests
pytest -v pySDC/tests
```

> :bell: Many components are tested (core, implementations, projects, tutorials, etc ...) which make the testing quite long.
> When working on a single part of the code, you can run only the corresponding part of the test by specifying the test path, for instance :
>
> ```bash
> pytest -v pySDC/tests/test_nodes.py  # only test nodes generation
> ```

## Code coverage

This stage allows to checks how much of the `pySDC` code is tested by the previous stage. It is based on the [coverage](https://pypi.org/project/coverage/) library and currently applied to the following directories :

- `pySDC/core`
- `pySDC/projects`
- `pySDC/tutorial`

This analysis is done in parallel to the test each time a pull is done on any branch (main repository or fork).
You can look at the current coverage report for the master branch [here](https://parallel-in-time.org/pySDC/coverage/index.html) or compare the results with previous builds [here](https://app.codecov.io/gh/Parallel-in-Time/pySDC). Codecov will also comment on any pull request, indicating the change of coverage.

During developments, you can also run the coverage tests locally, using :

```bash
echo "print('Loading sitecustomize.py...');import coverage;coverage.process_startup()" > sitecustomize.py
coverage run -m pytest --continue-on-collection-errors -v --durations=0 pySDC/tests
```

> :bell: Note that this will run all `pySDC` tests while analyzing coverage, hence requires all packages installed for the [code testing stage](#code-testing).

Once the test are finished, you can collect and post-process coverage result :

```bash
coverage combine
python -m coverage html
```

This will generate the coverage report in a `htmlcov` folder, and you can open the `index.html` file within using your favorite browser.

> :warning: Coverage can be lower if some tests fails (for instance, if you did not install all required python package to run all the tests).

### Coverage exceptions

Some types of code lines will be ignored by the coverage analysis (_e.g_ lines starting with `raise`, ...), see the `[tool.coverage.report]` section in `pyproject.toml`.
Part of code (functions, conditionaly, for loops, etc ...) can be ignored by coverage analysis using the `# pragma: no cover`, for instance

```python
# ...
# code analyzed by coverage
# ...
if condition:  # pragma: no cover
    # code ignored by coverage
# ...
# code analyzed by coverage
# ...
def function():  # pragma: no cover
    # all function code is ignored by coverage
```

Accepted use of the `# pragma: no cover` are:

1. Functions and code used for plotting
2. Lines in one conditional preceding any `raise` statement

If you think the pragma should be used in other parts of your pull request, please indicate (and justify) this in your description.

## Documentation generation

Documentation is built using [sphinx](https://www.sphinx-doc.org/en/master/).
To check its generation, you can wait for all the CI tasks to download the `docs` artifacts, unzip it and open the `index.html` file there with you favorite browser.

However, when you are working on documentation (of the project, of the code, etc ...), you can already build and check the website locally :

```bash
# Run all tests, continuing even with errors
pytest --continue-on-collection-errors -v --durations=0 pySDC/tests
# Generate rst files for sphinx
./docs/update_apidocs.sh
# Generate html documentation
sphinx-build -b html docs/source docs/build/html
```

Then you can open `docs/build/html/index.html` using you favorite browser and check how your own documentation looks like on the website.

> :bell: **Important** : running all the tests is necessary to generate graphs and images used by the website.
> But you can still generate the website without it: just all images for the tutorials, projects and playgrounds will be missing.
> This approach can be considered for local testing of your contribution when it does not concern parts containing images (_i.e_ project or code documentation).

:arrow_left: [Back to Pull Request Recommendation](./01_pull_requests.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to Naming Conventions](./03_naming_conventions.md)
