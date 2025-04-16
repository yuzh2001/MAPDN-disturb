
first, install uv.

```
uv sync

uv run src/test/test.py --save-path ./packages/mapdn/trial/model_save --alg matd3 --alias 0 --scenario case33_3min_final --voltage-barrier-type l1 --test-mode single --test-day 730 --render
```