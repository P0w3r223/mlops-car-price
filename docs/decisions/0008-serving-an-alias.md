# ADR 0008 — The service resolves an alias, and starts degraded rather than not at all

Date: 2026-07-22
Status: accepted
Author: P0w3r223 + Claude
Related to: ADR 0001, ADR 0003

---

## Context

Project A3 serves a model file baked into its container image: the artifact is copied in at
build time and loaded from disk at startup. That is the simplest thing that works, and it
ties two lifecycles together — a new model means a new image, a new deploy, and a code
repository that has to know which model is current.

This project already has a registry with a movable `champion` alias, so the serving layer has
a choice about how much it needs to know.

A second question comes with it. What should the service do when there is no champion — a
fresh environment, a rolled-back promotion, an unreachable tracking server?

## Options

**Where the model comes from**

1. **Bake the artifact into the image**, as A3 does. No runtime dependency on the registry,
   and the deployed artifact is exactly what was tested. Every promotion becomes a rebuild,
   and "which model is running?" is answered by reading a Dockerfile.
2. **Resolve `models:/car-price@champion` at startup.** The image contains no model and never
   changes when the model does; a promotion becomes a restart. The registry is now on the
   critical path for starting up.
3. **Resolve on every request.** Always current, and pays deserialisation on the hot path
   while turning every prediction into a dependency on the registry being up.

**What to do with no champion**

A. **Refuse to start.** Loud, and in an orchestrator it produces a crash loop whose cause is
   buried in logs.
B. **Start, report `degraded`, refuse predictions with 503.** The container runs, readiness
   fails so no traffic is routed, and `/health` and `/model-info` can be asked why.

## Decision

Option 2 with behaviour B. The champion is resolved once at startup; `/health` reports
`degraded` with a reason when there is none, and `/predict` answers 503 rather than pretending.

## Consequences

- **Deployment and promotion are separated.** Moving the alias and restarting the service is
  the whole deployment procedure; the image is unchanged and so is this repository.
- **The registry is a startup dependency**, which is the price of that separation. A service
  that is already running keeps serving if the tracking server goes away — the model is in
  memory — but it cannot start.
- **Every answer carries its provenance.** `X-Model-Version` on the response and `/model-info`
  make "which model priced this car?" answerable after the alias has moved on.
- **A no-champion environment is a normal state, not a failure.** It is exactly what a fresh
  deployment looks like before the first promotion, and the tests assert it.
- **Restarting is the only way to pick up a new champion.** Deliberate: a hot-reload endpoint
  would make the serving version change under a running experiment. Tracked as an issue
  rather than smuggled in.
- **The tracking URI is environment-driven**, so the same image talks to a local SQLite store
  in development and to a tracking server in compose. That also forced a fix: an experiment's
  artifact location must not be pinned to a client-side path when a server owns the artifacts.
