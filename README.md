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


A high-performance binning library specifically designed for **Credit Risk Modeling** and **Scorecard Development**. 

In financial risk modeling, **Weight of Evidence (WoE)** and **Information Value (IV)** are gold standards for feature engineering. `fastbinning` ensures mathematical rigor with extreme speed.

# Why fastbinning for Credit Scoring?

* **Monotonicity Guaranteed**: In credit scoring, features like 'Utilization Rate' or 'Age' must have a monotonic relationship with default risk to be explainable and compliant.
* **Built for Big Data**: While traditional tools struggle with millions of rows, `fastbinning` handles 10M+ records in milliseconds.
* **Robustness**: Prevents overfitting by enforcing minimum sample constraints (`min_bin_pct`), ensuring each bin is statistically significant.

# Installation
Install using pip:
```bash
pip install fastbinning
```

# Example
Please refer to the [**Examples**](https://github.com/RektPunk/fastbinning/tree/main/examples) provided for further clarification.
