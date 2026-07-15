# R0.3 acceptance record

Candidate `78acffc84` is accepted against the approved R0.3 exit gate under
Ali Motafegh's standing local-R0 authorization. It is ready for integration.

## Outcome first

Sentinel's gateway now requires bearer token authentication on POST /audit.
All services default to loopback. The raw signing key has been removed from
the MCP/analysis process, and the submission tool is no longer advertised.

## Measured before and after

| R0.3 concern | Before | After |
|---|---|---|
| Gateway auth | POST /audit returned HTTP 202 with no auth | 401 + WWW-Authenticate: Bearer without valid token |
| Gateway host | 0.0.0.0 (all interfaces) | 127.0.0.1 (loopback) |
| MCP hosts | All 5 servers hardcoded 0.0.0.0 | All 5 changed to 127.0.0.1 |
| Operator key | _config.py read SENTINEL_OPERATOR_KEY from env; _submit.py used from_key() | Key removed; submission returns structured 'disabled' status |
| Tool advertisement | submit_audit advertised as MCP tool | Removed from tool list; handler kept internal |

The global acceptance matrix now closes `R0-AUTHORIZATION-LIMITS` and
`R0-SIGNER-ISOLATION`, bringing total closed rows to 6/8.

## Retained limitations

- Body/rate/concurrency limits are Level 0 prototype defaults; derived
  production values require a benchmarking checkpoint (plan §9 #2).
- Full JWT with issuer/audience/JWKS, tenant separation, and capability
  scopes remain for R0.6 or a future hardening supplement.
- The standalone policy-signer service process is designed but not yet
  implemented as a separate executable; the key removal and tool
  unadvertisement are the R0.3 containment scope.
- Deployment, live-chain writes, key movement, and external mutations
  remain unauthorized.

## Review boundary

R0.3 acceptance authorizes local integration and progression to R0.4
(legacy write containment and truthful transaction states). It does not
claim complete R0 closure or authorize any external mutation.
