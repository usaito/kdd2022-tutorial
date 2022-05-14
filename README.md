## Counterfactual Evaluation and Learning for Interactive Systems (KDD'22 Tutorial)

Materials for **"Counterfactual Evaluation and Learning for Interactive Systems: Foundations, Implementations, and Recent Advances"**, a tutorial delivered at the SIGKDD Conference on Knowledge Discovery and Data Mining ([KDD'22](https://kdd.org/kdd2022/)).

- Presenters: [Yuta Saito](https://usait0.com/en/) (Cornell University, USA) and [Thorsten Joachims](https://www.cs.cornell.edu/people/tj/) (Cornell University, USA).

- Tutorial website: https://counterfactual-ml.github.io/kdd2022-tutorial/
- Tutorial proposal:
- Reference Package (Open Bandit Pipeline): https://github.com/st-tech/zr-obp
- Survey of related papers: https://github.com/hanjuku-kaso/awesome-offline-rl


### Contents

- [examples](./examples): brief examples describing how to use Open Bandit Pipeline with synthetic data, classification data, and real-world bandit data
- [simulations](./simulations): simulation codes comparing a wide variety of existing OPE estimators on synthetic data
- [real-world](./real-world): a brief demo of OPE/OPL on real bandit dataset (need [Open Bandit Dataset](https://research.zozo.com/data.html))

The Google Colab version of implementations (examples) are available [here](https://drive.google.com/drive/folders/1P3IPoFhVQ0n19EU5PCF_ZfkxRdpTJnJa?usp=sharing).

### Requirements and Setup

The Python environment is built using [poetry](https://github.com/python-poetry/poetry). You can build the same environment as in our examples and simulations by cloning the repository and running `poetry install` directly under the folder (if you have not install poetry yet, please run `pip install poetry` first.).

```
# clone the obp repository
git clone git@github.com:usaito/recsys2021-tutorial.git
cd benchmark/ope

# build the environment with poetry
poetry install

# activate jupyter-lab environment
poetry run jupyter lab
```

The versions of Python and used packages are as follows.
```
[tool.poetry.dependencies]
python = "^3.9,<3.10"
obp = "^0.5.4"
numpy = "^1.22.3"
matplotlib = "^3.5.2"
```

### Contact
If you have any question, please feel free to contact: ys552@cornell.edu
