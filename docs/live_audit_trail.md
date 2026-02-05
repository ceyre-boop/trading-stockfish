# Live Audit Trail Contract (Phase 11)

Scope: Capture every significant live event with lossless, append-only semantics.

Events to log
- Orders (submit/cancel/replace)
- Fills (partial/full)
- SAFE_MODE events
- Guardrail events
- Anomalies
- Policy references (version, hash)
- Connector health events

Implementation requirements
- engine/live_audit_writer.py writes JSONL entries, schema-validated.
- Append-only writes with durability checks; integrates with storage layer.
- Timestamps and correlation IDs required.

Acceptance
- Audit trail must be complete and lossless; every execution path writes an auditable record.
