# hmm-slr

To run the experiments, simply open the `runExperiments.m` file to view and
change parameters and then run the script. It will train N models using 2
training signs each. It will then test all of the trained models using the test
set.

Please note that models may not train properly due to poor initial probabilities.
This causes MATLAB to report warnings due to singular matrices. You will have to
stop the script and retry. I did my best to look into why this was happening,
but could not find a good solution.
