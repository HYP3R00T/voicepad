# Security Policy

## Supported versions

The current `main` branch is the supported development version. Published package support follows the latest reviewed PyPI release; older versions may not receive fixes.

## Report a vulnerability

Do not disclose suspected vulnerabilities in a public issue, discussion, pull request, log, or recording attachment.

Use GitHub private vulnerability reporting from the repository's **Security** page when available. Otherwise email [hyperoot.tech@proton.me](mailto:hyperoot.tech@proton.me) with a concise description and request a private channel before sending sensitive material.

Include, when safe and relevant:

- affected version or commit;
- impact and affected users;
- minimal reproduction steps;
- relevant operating system and configuration;
- known mitigations; and
- whether the issue has been disclosed elsewhere.

Remove credentials, tokens, private recordings, transcripts, personal information, and unrelated logs. A proof of concept is welcome but not required.

## Response

The maintainer will make a reasonable effort to acknowledge the report, assess impact, and coordinate remediation or disclosure. No fixed response or remediation timeline is guaranteed.

Keep details confidential until a fix, mitigation, or coordinated disclosure plan is available. Accepted reports may be managed through a private GitHub security advisory.

## Security releases

A security fix that requires package publication follows the same protected release-PR workflow as any other release. Prepare and review repository release state before publication, merge it explicitly, and let the post-merge job publish from that reviewed commit. Do not bypass branch protection, publish from an ordinary issue pull request, or add, expose, or change publishing credentials as part of an unrelated fix.
