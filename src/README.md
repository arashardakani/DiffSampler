
# Installation

```[bash]
    # in your conda/virtualenv; assuming CUDA v.12
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install numpy
    pip install tqdm
    pip install python-sat
    pip install wandb
```

# run experiment reproduction

```[bash]
# from repo top-level
./scripts/run_exp_or.sh adam/sgd
./scripts/run_exp_blasted.sh adam/sgd
```
Instructions are same across all benchmark subsets (or, blasted, tire, prod, modexp, hash).

