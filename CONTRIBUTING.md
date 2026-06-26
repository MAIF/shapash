# How to contribute to Shapash Open source

This guide aims to help you contributing to Shapash. If you have found any problems, improvements that can be done, or you have a burning desire to develop new features for Shapash, please make sure to follow the steps bellow.

- [How to open an issue](#how-to-open-an-issue)
- [Create your contribution to submit a pull request](#create-your-contribution-to-submit-a-pull-request)
    - [Fork to code in your personal Shapash repo](#fork-to-code-in-your-personal-shapash-repo)
    - [Clone your forked repository](#clone-your-forked-repository)
    - [Make sure that your repository is up to date](#make-sure-that-your-repository-is-up-to-date)
    - [Start your contribution code](#start-your-contribution-code)
    - [Commit your changes](#commit-your-changes)
    - [Create a pull request](#create-a-pull-request)
    - [Finally submit your pull request](#finally-submit-your-pull-request)

# How to open an issue

**Screenshots are coming soon**

An issue will open a discussion to evaluate if the problem / feature that you submit is eligible, and legitimate for Shapash.

Check on the project tab if your issue / feature is not already created. In this tab, you will find the roadmap of Shapash.

A Pull Request must be linked to an issue.
Before you open an issue, please check the current opened issues to ensure there are no duplicate. Define if it's a feature or a bugfix.

Next, the Shapash team, or the community, will give you a feedback on whether your issue must be implemented in Shapash, or if it can be resolved easily without a pull request.

# Create your contribution to submit a pull request
## Fork to code in your personal Shapash repo

The first step is to get our MAIF repository on your personal GitHub repositories. To do so, use the "Fork" button.

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-fork.png" alt="fork this repository" />

## Clone your forked repository

<img align="right" width="300" src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-clone.png" alt="clone your forked repository" />

Click on the "Code" button to copy the url of your repository, and next, you can paste this url to clone your forked repository.

```
git clone https://github.com/YOUR_GITHUB_PROFILE/shapash.git
```

## Make sure that your repository is up to date

To insure that your local forked repository is synced, you have to update your repo with the master branch of Shapash (MAIF). So, go to your repository and as follow :

```
cd shapash
git remote add upstream https://github.com/MAIF/shapash.git
git pull upstream master
```

## Install dependencies and setup your virtual environment
Python `>=3.11, <3.15` is required.

After creating a virtual environment, run
```
pip install -e ".[dev,test-full]"
```

Otherwise, we recommend to use `uv`
```
uv sync --extra dev --extra test-full
```
Take a look at `pyproject.toml` for other dependency groups.

## Pre-commit
We use pre-commit hooks to automatically identify issues before committing. After installing dev dependencies, make sure to set up the git hook scripts to run automatically on `git commit`
```
pre-commit install
```

## Start your contribution code

To contribute to Shapash, you will need to create a personal branch.
```
git checkout -b feature/my-contribution-branch
```
We recommand to use a convention of naming branch.
- **feature/your_feature_name** if you are creating a feature
- **hotfix/your_bug_fix** if you are fixing a bug

## Tests & Build

Run pytest to check that all tests pass (and get coverage):
```
pytest --cov=shapash
```

To test against multiple Python versions at once, use `tox`:
```
pipx install tox
pipx inject tox tox-uv
tox
```
`tox-uv` lets tox use `uv` for interpreter and environment management. Missing Python versions are skipped automatically.

Make sure that Shapash builds correctly:
```
python -m build

or

uv build
```

## Code Quality
Check your code quality (linting and formatting) with ruff:
```
ruff check
ruff format
```

## Commit your changes

We recommend committing with clear messages and grouping your commits by modifications dependencies.

Once all of the previous steps succeed (code quality, tests and build), push your local modifications to your remote repository.

```
git add <file>
git commit -m 'fixed a bug'
git push origin feature/my-contribution-branch
```

Your branch is now available on your remote forked repository, with your changes.

Next step is now to create a Pull Request so the Shapash Team can add your changes to the official repository.

## Create a Pull Request


A pull request allows you to ask the Shapash team to review your changes, and merge your changes into the master branch of the official repository.

To create one, on the top of your forked repository, you will find a button "Compare & pull request"

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-compare-pr.png" alt="pull request" />

As you can see, you can select on the right side which branch of your forked repository you want to associate to the pull request.

On the left side, you will find the official Shapash repository.

- Base repository: MAIF/shapash
- Base branch: master
- Head repository: your-github-username/shapash
- Head branch: your-contribution-branch

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-pr-branch.png" alt="clone your forked repository" />

Once you have selected the right branch, let's create the pull request with the green button "Create pull request".

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-pr-description.png" alt="clone your forked repository" />

In the description, a template is initialized with all informations you have to give about what you are doing on what your PR is doing.

Please follow this to write your PR content.


## Finally submit your pull request

Your pull request is now ready to be submitted. A member of the Shapash team will contact you and will review your code and contact you if needed.

You have contributed to an Open source project, thank you and congratulations ! 🥳

Show your contribution to Shapash in your curriculum, and share it on your social media. Be proud of yourself, you gave some code lines to the entire world !
