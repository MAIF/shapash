# Security Policy

## Supported versions

Security fixes are made on the latest minor release only. Please upgrade before reporting.

## Reporting a vulnerability

**Please do not open a public issue, pull request or discussion for a security problem.**

Report it privately through GitHub:
[Report a vulnerability](https://github.com/MAIF/shapash/security/advisories/new)
(the *Security* tab of the repository → *Report a vulnerability*).

Please include:

- the affected version(s) of Shapash and of the relevant dependency (Dash, transformers, …);
- a description of the issue and its impact;
- the steps or a minimal script to reproduce it;
- a suggested fix, if you have one.

## What to expect

Shapash is maintained with limited time, so we cannot guarantee response or fix times. We will
review every report as soon as we can and keep you informed of our assessment.

For confirmed issues, we publish a patch release and a GitHub Security Advisory, and request a CVE
when relevant. Reporters are credited unless they ask not to be.

Please keep the details private until a fix is released.

## Security model

Shapash is a library that runs with the privileges of the Python process that uses it. Some of its
features are unsafe by design when given untrusted input; these are documented behaviours, not
vulnerabilities:

- **Loading saved explainers.** `SmartExplainer.load()` and `load_smartpredictor()` use `pickle`.
  Unpickling a file can run arbitrary code: only load files you created or fully trust. Their type
  checks run after unpickling, so they offer no protection. `NlpExplanation.load()` reads a
  zip of JSON and Parquet files and must not execute code.
- **Loading models.** Loading a Hugging Face checkpoint with `trust_remote_code=True` runs code
  shipped with that checkpoint. Shapash keeps it `False` by default.
- **The webapp.** `run_app()` starts a Dash development server with no authentication, bound to
  `127.0.0.1` by default. It is meant for local exploration. Binding it to a public interface
  exposes your model and data to anyone who can reach it; for shared use, put it behind a
  production WSGI server and an authenticating reverse proxy. Never enable `debug=True` on a
  network-reachable server.
- **HTML reports** embed the data and explanations they were generated from. Treat them as
  being as sensitive as that data.

Reports that a bug in one of these areas breaks the guarantees above (for example, the webapp
reachable from another machine with default settings, or code executed when loading a file
format that is meant to be safe) are in scope.

Vulnerabilities in dependencies should be reported to those projects. If Shapash needs a new
minimum version of a dependency because of one, a regular issue or pull request is fine.
