# Support

## Usage and documentation

Start with:

- [README](README.md)
- [Deployment guide](docs/deployment.md)
- [Portal and request-mode guide](docs/portal-mode.md)
- [Security model](docs/security-model.md)
- [Endpoint coverage](docs/endpoint-coverage.md)
- [Maintenance playbooks](docs/MAINTENANCE_PLAYBOOKS.md)

For a reproducible non-security bug, open a GitHub issue using the appropriate template. Include the
package version or image digest, runtime and credential mode, sanitized health fields, expected
behavior, and the smallest provider-free reproduction available.

Feature requests should explain the agent workflow, intended Qdrant operation, required permissions,
output bounds, and why existing typed tools do not cover the use case.

## Not appropriate for public issues

Do not post credentials, tenant identifiers, collection contents, database exports, private endpoint
names, runtime files, deployment evidence, or uncoordinated vulnerability details. Use the private
path in [SECURITY.md](SECURITY.md) for security reports.

This project does not provide Qdrant Cloud account support, managed hosting, data recovery, or a
guaranteed support SLA.
