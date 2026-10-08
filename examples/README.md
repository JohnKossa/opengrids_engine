# Examples

All of these run on the bundled synthetic dataset (`opengrids.demo`), so they
work out of the box: `pip install -e .` then `python examples/<file>.py`.

| file | shows |
|---|---|
| `quickstart.py` | a config, a data check, one grid, rates, metrics and the peel trace |
| `from_model.py` | starting from a fitted scikit-learn model (here a log-price MRA) instead of SHAP values |
| `change_adjustment_fields.py` | choosing which fields are adjusted, engine-wide or for one grid; how that changes the comps needed |
| `add_unmodeled_field.py` | adjusting for a field the model doesn't have (Frisch-Waugh-Lovell), and how pool size affects it |
| `assessor_overrides.py` | per-search and office-wide override rates, categorical variant overrides, overriding an FWL field, overrides vs. the sign clamp |

To use your own data, replace `make_demo()` with `Dataset.from_files(...)` or
`Dataset.from_model(...)` (see the main README).
