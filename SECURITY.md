# Security Policy

## Supported versions

Starlibpy is currently in public beta. Security and scientific-correctness
fixes are prioritized for the latest published `0.3.x` beta release.

## What to report privately

Please report privately:

- a software vulnerability;
- unsafe deserialization, path handling, or file export behavior;
- dependency or supply-chain compromise;
- exposure of sensitive data;
- a defect that can silently change statistical estimates, confidence
  intervals, p-values, classifications, survival results, or clinical outputs;
- a reproducible issue that may materially affect published research.

Ordinary usage questions and non-sensitive feature requests may be opened as
public issues.

## How to report

After the GitHub repository is public, use GitHub's private vulnerability
reporting or a private repository security advisory. Until then, contact the
maintainer privately through the contact method listed on the maintainer's
GitHub profile.

Include:

- affected Starlibpy version;
- Python version and operating system;
- affected function or module;
- minimal reproducible example;
- expected and observed behavior;
- potential impact;
- any suggested mitigation;
- whether the report may be publicly credited.

Do not include real patient or participant data.

## Coordinated disclosure

Please allow reasonable time to investigate, test, and prepare a fix before
public disclosure. The project will acknowledge receipt, assess severity,
communicate progress when possible, and credit reporters who request
attribution.

## Scientific disclaimer

Security review does not replace independent statistical validation, clinical
judgment, protocol-specific adjudication, or regulatory review.
