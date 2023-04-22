### Installation
Clone the repository, activate an environment with Python 3.10, and run 
```bash
pip install -e .
```
Since JAX requires `--find-links` which is not supported by `pyproject.toml` to the best of our knowledge for installation, we hardcode the location of the `jaxlib` wheels. If you need to work with a different Python or CUDA version, feel free to edit the dependencies or install JAX by [the official instrutions](https://github.com/google/jax#installation).
Although you can use `venv` to manage your environments, if you encounter any problem with CUDA, try to use a conda environment and run
```bash
conda install cuda -c nvidia
```

### Run Experiments
Add your config to `src/d3exp/config`, and run 
```bash
python -m d3exp ${CONFIG_NAME}
```