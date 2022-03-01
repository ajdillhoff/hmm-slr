# hmm-slr

This was my final project for a graduate machine learning course in 2015.
I was interested in implementing HMMs and EM for American Sign Language recognition, using only one-handed signs for simplicity.
I based the code off of [Rabiner's tutorial paper](https://courses.physics.illinois.edu/ece417/fa2017/rabiner89.pdf).

To run the experiments, simply open the `runExperiments.m` file to view and
change parameters and then run the script. It will train N models using 2
training signs each. It will then test all of the trained models using the test
set.

Please note that models may not train properly due to poor initial probabilities.
This causes MATLAB to report warnings due to singular matrices. You will have to
stop the script and retry. I did my best to look into why this was happening,
but could not find a good solution.
