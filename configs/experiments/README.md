# Experiment overrides

Small YAML files that override a subset of [../parameters.yaml](../parameters.yaml). They are
deep-merged on top of it: `load_parameters(overrides=["<name>"])`, or
`load_filter_config("<name>")` for the filter section. List only what changes, each with a
comment saying why. Unknown keys are rejected.

Example (`no_nhc.yaml`):

```yaml
filter:
  name: eskf-baseline-no-nhc
  nonholonomic:
    enabled: false   # scope-literal baseline without the non-holonomic constraint
```
