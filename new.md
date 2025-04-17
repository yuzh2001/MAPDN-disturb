
first, install uv.

```bash
uv sync
uv pip install -e .
apt install freeglut3-dev python3-opengl -y

xvfb-run -s "-screen 0 1400x900x24" uv run src/test/test.py --save-path reproduction/model_save --alg mappo --alias 0 --scenario case33_3min_final --voltage-barrier-type bowl --test-mode single --test-day 730 --render
```