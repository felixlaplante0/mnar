# Presentation

This package implements 4 different but closely related models for MCAR/MNAR modeling. The modes are as follows: `mode="r"` (MCAR with constraint of equality across variables), `mode="rj"` (MCAR), `mode="z"` (MNAR$z$) and `mode="zj"` (MNAR$z^j$). Note there is an automated CI with strict linting rules enforcing PEP-8 compliance. A native `scikit-learn` integration is provided, ensuring maximum quality of the code and ease of use. The code is highly efficient and optimized for large scale datasets with moderately low dimensional data.

# Installation

For now, the installation is only local, as the package has not yet been submitted to the PyPI repository, but this will be shortly remedied. To install the required packages, please run: `pip install fastkmeanspp>=0.2.0 scikit-learn numpy scipy`.

# Scripts

The scripts yield reproducible results (except for the execution times). The mains scripts are `simulation-plots.py` for the plots on simulated data (ARI, MSE, execution times), and `iris-plots.py` which is its counterpart for tests on a real dataset (Iris). The other files are generating pictures for the Monte-Carlo verification of the asymptotic ARI formula ($ARI \overset{\text{a.s.}}{\to} \left( 1 - 2 \Phi(-\frac{d}{2 \sigma}) \right)^2$) and the other to illustrate the padding used to vectorize the implementation.
