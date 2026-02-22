GEMINI.md : lnm-toolkit development guide
Todo: 
- [ ] Use the saved "permuted_indices.npy" created during permuting the GLM with PRISM to also run permutations on the sensitivity analysis, and save those results as well. This will allow use to calculate false positive rates for the different analyses and compare them directly, aligned with the same permuted indices.
- [ ] Add a function to calculate and save the false positive rates for both the GLM and sensitivity analyses, using the saved permuted.
- [ ] Create a cli tool for running our analyses without need for a notebook or script. Just a simple CLI. 
- [ ] Make sure all docstrings are up to date, clear and consistent, keeping in mind we use these for automated docs generation with mkdocs.
- [ ] Add more unit tests for the different analyses functions, including edge cases and error handling. We're gonna need to be brutal with our tests to make sure we catch any potential issues before they arise in real data analyses.
- [ ] Create a README.md file with clear explanation of the point of the toolkit, what it does, how to use it, and the docs link.
- [ ] Update the landing page to better reflect the purpose of the toolkit and link to the README and docs.