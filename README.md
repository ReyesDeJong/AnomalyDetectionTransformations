# Deep Anomaly Detection Using Geometric Transformations
To be presented in NIPS 2018 by Izhak Golan and Ran El-Yaniv.

## Introduction
This is the official implementation of "Deep Anomaly Detection Using Geometric Transformations".
It includes all experiments reported in the paper.

## Requirements
* Python 3.5+
* Keras 2.2.0
* Tensorflow 1.8.0
* sklearn 0.19.1

## Citation
If you use the ideas or method presented in the paper, please cite:

```
@article{golan2018deep,
  title={Deep Anomaly Detection Using Geometric Transformations},
  author={Golan, Izhak and El-Yaniv, Ran},
  journal={arXiv preprint arXiv:1805.10917},
  year={2018}
}
```

## Version Specifications 0.1.2

r0.1.2 Before including fixes to dirichlet bug; trivial constant transform ruined alpha_0 init
r0.1.1 Before modification to DeepHits.py. Future will include modifications to predict, to avoid memry leak
r0.1.0 It contain thesis work before start modification of Discrimination Matrix Method, which is contained in scripts/transformation_clean/training_transformation_selection.py