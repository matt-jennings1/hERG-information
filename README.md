# hERG-information
Code for generating synthetic IKr curves for a range of voltage protocols, and subsequently deriving information-theoretic metrics used to appraise their efficacy. Some results are included, but not all of the raw data (e.g. simulated current curves) due to file size considerations.

# PRE-REQUISITES

The majority of the code is written in Python 3, and requires `numpy`, `scipy`, and `scikit-learn`. `matplotlib` is used for producing figures. Additionally, `NPEET` (https://github.com/gregversteeg/NPEET) was used for its implementation of the CMI estimator described in the paper.

In order to simulate your own IKr curves, a C compiler (such as GCC) and SUNDIALS 2.4.0 (https://computing.llnl.gov/projects/sundials) are also required. ODE solving packages for Python may be used instead, but we found CVODE to have a significant advantage for the stiffer parameter samples.

These instructions primarily consider Linux users; users of a different OS will have to adapt `run_model.py` to run their executable ODE solver, or opt to use one included in a Python package.

# Usage guidance

In its current state, the framework requires the execution of several programs, rather than one large program/script. A step-by-step guide to each of the programs is provided:

1. Run `gen_params.py`, specifying the desired number of samples and directory name with the `n_desired` and `tag` variables, to randomly sample parameter values from the prior. By default, 10,000 samples will be created in `./output/10000_Alpha`. The file structure creates a subfolder for each sample within this directory, wherein the parameter samples and each of the simulated IKr curves will be saved.

2. Compile `HHFullOut` using the provided Makefile (you may have to edit this to link to your SUNDIALS install directory. Run `run_model.py`, which by default will simulate results for all seven protocols (this will take a long time!). Since each CVODE instance runs on one core, this is easily parallelisable by modifying the code so that there are multiple executables, and multiple input/output file names to prevent conflicts

**Note:** Typically, the solver returns an error for a handful of parameters in each set of 10,000. This corresponds with extreme or 'unrealistic' values being sampled for that subset of parameters. It is necessary to re-sample the parameters and re-run the model for all protocols for those re-sampled subsets, to ensure all protocols run on the sample dataset. For small numbers of errors this can easily and quickly be done manually, but if for whatever reason a large number of errors have occurred (such as by altering the prior or protocol), `gen_params.py` can be modified to accomplish this by iterating through a list of failed samples. Should any of the new samples also fail, they will have to be re-sampled once more.

3. Next, use `PCA_DR.py` to apply the PC decomposition and dimensionality reduction. This will save the PCs (modes), and the PC scores for each sample.

4. To transform the data into its optimal form, first run `combine_params_and_output.py` to combine each parameter sample subset with its respective PC scores. Then, run `alpha_transform.py` to transform the parameter set, and then run `PC_transform.py` to transform the PC scores.

5. Finally, the data is now ready for the (C)MI estimators, which are implemented in `MI_estimate.py` and `CMI_estimate.py`. Once MI estimates are obtained, EEV can be calculated through the use of `EEV.py`. The primary criterion, S4, is simply the maximum EEV for each protocol returned by this function. Should the other *S* criteria be desired as well, these can be obtained through `Scriterion.py`.

