# BQTmizer: A Tool for Test Case Minimization with Quantum Annealing

## Description
Our innovative tool harnesses the power of quantum annealing (QA) to address large-scale real-world test case minimization (TCM) challenges. BQTmizer is designed to select the smallest possible test suite while ensuring that all testing objectives are met.

As a hybrid solution, it integrates bootstrap sampling techniques to optimize qubit usage in QA hardware. BQTmizer enhances your testing process, making it faster, smarter, and more efficient.

## Installation
```
pip install bqtmizer
```

## Example Usage
The dataset of the original test suite should be provided by a CSV file, where each column represents a property of test case and each row represents a test case.

Effectiveness and cost properties should be specified.

Users can create an account in the D-Wave Leap platform: [https://cloud.dwavesys.com/](https://cloud.dwavesys.com/) and paste the token here to run quantum annealing.

Users should specify the weights for each property, including for the objective of minimizing the number of test cases, using the key "num".

Users can specify the subproblem size N and coverage percentage beta.
```
from bqtmizer import bqtmizer

weights_dict = {"time":1/3, "cost":1/3, "num":1/3}
result = bqtmizer("../dataset/PaintControl_TCM.csv", ["rate"], ["time"], "xxx", weights=weights_dict, N=30, beta=0.9)
```