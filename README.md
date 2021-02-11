## commsim: spatial community clustering simulation

This program simulates spatial community assembly for use in downstream
community clustering and SDM-type algorithms 
(e.g., Maxent or Bayesian logistic regression)


- Model assumptions:
    - assemblages are composed of species with overlapping geographic ranges.
    - a species potential niche represents its abiotic constraints (n-dimensional).
    - a species realized niche is a subset of its potential niche due to competition.
    - a species geographic range is bounded by its realized niche, but with dispersal limits.  

- Model features:
    - a geographic area matrix (n x m).
    - a complete float matrix of environmental data (nvariables x n x m)
    - a sparse binary matrix of observed species occurrences (nspecies x n x m).   

- Model goals:
	- jointly infer:
		- species occurrence covariance matrix (n\*\*2)
		- species niche thresholds (nvar * n)
		- species geographic center of origin (n)
		- species decay parameters (n)
		- total: 2n + n \* nvar + n\*\*2
	- Use inferred parameters for posterior predictive fit to sparse occurrence data.
	- Use inferred parameters to interpolate species across unsampled grid cells.


- Inference methods:
    - Gaussian Processes to infer covariance matrix from occurrences + env variables.
    - Logistic Regression to infer decay parameters and origin.
    - Performed together in Pymc3 Bayesian model.


- Generative data:
    - Generate environment.
    - Get species potential niche.
    - Get species realized niche (geographic range) by comp. exclusion
