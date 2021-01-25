# How to open a GitHub issue & file a bug report

Before opening a pull request to contribute, we prefer that you create an issue in first time.

An issue will open a discussion to evaluate if the problem / feature that you identify is elligible, and legitim for Shapash.

A Pull Request must be linked with an issue.
Before you open your issue, please check what issues are already open to don't duplicate issues. Define if it's a feature or a bugfix.

Next, Shapash team, or community, will answer you to see if your issue must be implemented in Shapash, or if it can be resolved easily without a pull request.

# How to contribute to Shapash Open source

This guide aims to help you to contribute to Shapash. If you found any problems, improvement to do, or you want to help us to develop features, you can help us to make Shapash better.

## 1. Fork to code in your personal Shapash repo

The first step is to get the repo to your personal github repositories. To do it, use this button.

<img src="https://raw.githubusercontent.com/MaxGdr/shapash/contributing/docs/assets/images/contributing/shapash-fork.png" alt="fork this repository" />

## 2. Clone your forked repository

<img align="right" width="300" src="https://raw.githubusercontent.com/MaxGdr/shapash/contributing/docs/assets/images/contributing/shapash-clone.png" alt="clone your forked repository" />

Click on the "Code" button to copy the url of your repository, and next, you can paste this url to clone your forked repository.

```
git clone https://github.com/YOUR_GITHUB_PROFILE/shapash.git
```

## 3. Be sure that your repository is updated

To be sure that your forked repo you have in local is good, you have to update your repo with the Shapash's master branch. So in your forked, do this :

```
cd shapash
git remote add upstream https://github.com/MAIF/shapash.git
git pull upstream master
```

## 4. Start your contribution's code

To contribute to Shapash, you will need to create your branch.
```
git checkout -b feature/my-contribution-branch
```
We recommand to use a convention of naming branch. 
- **feature/your_feature_name** if your are creating a feature
- **hotfix/your_bug_fix** if want to fix a bug

## 5. Commit your changes

Once you have test your code in local, push it to your repository. 

When you did your changes, before commit, we have some recommendations :
- Execute pytests to check that all tests pass (Simply execute pytest)
```
pytest
```
- Try to build shapash 
```
python setup.py bdist_wheel
```
- Check your code with **flake8**

*We will soon add **pre commit** to check your code quality automatically on commit*
```
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```
If all of these seems like good, let's push it ! 
```
git add .
git commit -m ‘fixed a bug’
git push origin feature/my-contribution-branch
```
In addition, we recommend to commit with clear messages and regroup your commits by modifications dependency. 

Your branch is now available on your remote forked repository, with your changes. 

Next step is now to create a Pull Request, to ask Shapash's Team to add your changes in the official repository.

## 6. Create a Pull Request


A pull request allows you to ask to the Shapash's team to review your changes, and merge your changes into the master branch of the official repository.

To create one, on the top of your forked repository, you will find a button "Compare & pull request"

<img src="https://raw.githubusercontent.com/MaxGdr/shapash/contributing/docs/assets/images/contributing/shapash-compare-pr.png" alt="pull request" />

As you can see, you can select on the right side which branch of your forked repo you want to associate to the pull request. 

In the left side, there is the official Shapash repo master branch. 

- Base repository: MAIF/shapash
- Base branch: master
- Head repository: your-github-username/shapash
- Head branch: your-contribution-branch

<img src="https://raw.githubusercontent.com/MaxGdr/shapash/contributing/docs/assets/images/contributing/shapash-pr-branch.png" alt="clone your forked repository" />

Once you selected the good branch, let's create the pull request with the green button "Create pull request".

<img src="https://raw.githubusercontent.com/MaxGdr/shapash/contributing/docs/assets/images/contributing/shapash-pr-description.png" alt="clone your forked repository" />

In the description, a template is initialized with all informations you have to give about what you are doing on your PR.

Please follow this to write your PR content.


## 7. Finally, create your pull request

Your pull request is now ready, you can create it, a member of the Shapash's team will contact you and will review your code.

You have contribute to an Open source project, congratulations ! 🥳

Show that you contribute to Shapash in your CV, and share it on your social medias. Be proud of you, you gave some code lines to the entire world !