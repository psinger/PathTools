PathTools
=========

A collection of tools and algorithms suitable to work with paths (e.g., navigational).

For installing the package, just call ```python setup.py install```.

Currently, this tool box consists of a framework for Markov chains. It is possible to fit trail corpora with varying order Markov chain models. The framework includes several advanced methods for determining the appropriate Markov chain order (maximum likelihood methods, information-theoretic methods, Bayesian inference and cross validation prediction) [1].

Furthermore, the framework includes a class for calculating semantic similarity between concepts in human navigational paths [2].

Finally, the provided framework is essential for the HypTrails approach presented in [3] and referenced in https://github.com/psinger/HypTrails.

For a basic introduction to the framework please refer to the test examples. For further questions please conduct the issues section or drop me an email!

If you use the code, please cite the corresponding publication.

[1] Philipp Singer, Denis Helic, Behnam Taraghi and Markus Strohmaier, 
Detecting Memory and Structure in Human Navigation Patterns Using Markov Chain Models of Varying Order,
PLoS ONE, vol 9(7), 2014

[2] Philipp Singer, Thomas Niebler, Markus Strohmaier and Andreas Hotho, 
Computing Semantic Relatedness from Human Navigational Paths: A Case Study on Wikipedia, 
International Journal on Semantic Web and Information Systems (IJSWIS), vol 9(4), 41-70, 2013

[3] Philipp Singer, Denis Helic, Andreas Hotho and Markus Strohmaier,
HypTrails: A Bayesian Approach for Comparing Hypotheses About Human Trails on the Web,
24th International World Wide Web Conference, Florence, Italy, 2015 (Best Paper Award) 

![Alt Text](https://zenodo.org/badge/4207/psinger/PathTools.png)
