# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.2.x | ✅ |
| 0.1.x | ❌ |

Only the latest `0.2.x` release is actively supported with security fixes.

## Reporting a Vulnerability

Please report security vulnerabilities using GitHub's private vulnerability
reporting, not a public issue:

[github.com/memstrand/memstrand/security/advisories/new](https://github.com/memstrand/memstrand/security/advisories/new)

You should receive an acknowledgment within a few days. Please include:

- The version (or commit) affected
- Steps to reproduce
- The potential impact

## Scope notes

memstrand runs as a **local daemon** (`memstrand serve`) intended for a
single operator's machine. Keep these in mind when evaluating or reporting
an issue:

- **Loopback by default.** The daemon binds to `127.0.0.1` unless
  `MEMSTRAND_HOST` is explicitly set to something else, and refuses to
  start on a non-loopback host unless `MEMSTRAND_TOKEN` is also set.
- **`MEMSTRAND_TOKEN`** is a single shared bearer token gating `/api`,
  `/mcp`, and `/v1`. There is no per-user auth or RBAC today (see the
  README roadmap) — treat the daemon as single-operator, and set a token
  before exposing it beyond loopback.
- **CSRF/DNS-rebinding guards**: mutating requests with a cross-origin
  `Origin` are rejected, and loopback requests with an untrusted `Host`
  header are rejected. These protect against a malicious webpage or
  rebound DNS name reaching the daemon from a browser — they are not a
  substitute for `MEMSTRAND_TOKEN` on a non-loopback deployment.
- **`/api/ingest` reads local filesystem paths** the caller supplies, with
  no separate path allowlist beyond the usual loopback/Origin/token
  checks. Don't expose the daemon to untrusted callers, and don't run it
  as a privileged user.
- **Owner-facing browse surfaces** (`memstrand list`/`show`, the web UI's
  Browse view, `GET /api/atoms`) intentionally show memories across *all*
  scopes — they're for the human operator, not for an agent. Set
  `MEMSTRAND_TOKEN` if the machine is shared or multi-user.

See [docs/cli.md](docs/cli.md#security-notes) for the full security notes,
and [docs/configuration.md](docs/configuration.md) for every
`MEMSTRAND_*` variable.
