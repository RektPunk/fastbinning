<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=fastbinning&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
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
| 1,000,000   | Execution Time     | **0.0265s**  | 1.1773s     | **44.38x** Faster   |
|             | Information Value  | 2.3131       | 2.3190s     | **99.74%** Fidelity |
| 10,000,000  | Execution Time     | **0.2523s**  | 16.5100s    | **65.44x** Faster   |
|             | Information Value  | 2.3091       | 2.3177s     | **99.63%** Fidelity |

Reproducibility: You can reproduce these results by running the [**script**](https://github.com/RektPunk/fastbinning/tree/main/examples/benchmark.py).
