# PE-Fin Lite Diagnostic-12 Checkpoint Audit v1 (docs mirror)

Passive, offline, zero-network audit of End-to-End checkpoints C01–C14.

- **Not** formal accuracy
- **Not** formula lock authorization
- Does **not** mutate raw Diagnostic-12 predictions
- Live output directory (future): `outputs/pe_fin_lite_diagnostic12_checkpoint_live_v1`
- Audit subdirectory: `final_checkpoint_audit_v1/`

After a live Diagnostic-12 run, mirror audit artifacts here via:

```bash
python scripts/run_pe_fin_lite_checkpoint_audit_v1.py \
  --run-dir outputs/pe_fin_lite_diagnostic12_checkpoint_live_v1 \
  --docs-mirror docs/pe_fin_lite_diagnostic12_checkpoint_audit_v1
```

Engineering-pass ≠ accuracy-pass. See `coverage_map.md` and package
`hk_equity_mcp.experiment.pe_fin_lite_checkpoint_audit`.
