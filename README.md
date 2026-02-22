<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=fastbinning&section=header&reversal=false&height=120&fontSize=90">
</div>
<p align="center">
  <a href="https://github.com/RektPunk/fastbinning/releases/latest">
    <img alt="release" src="https://img.shields.io/github/v/release/RektPunk/fastbinning.svg">
  </a>
  <a href="https://github.com/RektPunk/fastbinning/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/RektPunk/fastbinning.svg">
  </a>
</p>


A high-performance **binning** library specifically designed for Credit Risk Modeling and Scorecard Development.

## Let's be honest: Binning is a pain. 

In Credit Risk Modeling, binning with millions of rows often feels like a bottleneck.  You need to ensure Monotonicity, handle Missing Values, and maximize IV—all while your script runs for minutes.

`fastbinning` was born to solve this.  It delivers the near optimal mathematical precision of optimal binning at speeds you've never experienced before.

## Why fastbinning?

* Monotonicity Guaranteed: No more manual tweaking. Automatically enforces a monotonic trend in Weight of Evidence (WoE) for numerical features.
* Built for the Impatient: Binning shouldn't be a coffee break. It processes 10M+ records in milliseconds.
* Near-Optimal: Achieves near-optimal IV fidelity compared to Mixed-Integer Linear Programming solvers.

## Installation
Install using pip:
```bash
pip install fastbinning
```

## Example
Please refer to the [**Examples**](https://github.com/RektPunk/fastbinning/tree/main/examples) provided for further clarification.

## Benchmark
We sacrifice little of Information Value to achieve nearly two orders of magnitude speed improvement.

| Sample Size | Metric             | fastbinning  | optbinning  | comparison          |
|-------------|--------------------|--------------|-------------|---------------------|
| 1,000,000   | Execution Time     | **0.0216s**   | 1.0197s     | **47.14x** Faster   |
|             | Information Value  | 2.3013       |  2.3190     | **99.24%** Fidelity |
| 10,000,000  | Execution Time     | **0.1817s**  | 13.4070s    | **73.79x** Faster   |
|             | Information Value  | 2.2990       | 2.3177      | **99.19%** Fidelity |

Reproducibility: You can reproduce these results by running the [**script**](https://github.com/RektPunk/fastbinning/tree/main/examples/benchmark.py).
