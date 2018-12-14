# Final Project for EECS 545 (Machine Learning) Fall 2018

Implement comparative analysis of TSA, VOpt and active SVM for MNIST data.

The code is structured as follows:

-   Loading Data:
    -   `mnist.py`: Load entire dataset. Obtained from https://github.com/hsjeong5/MNIST-for-Numpy
    -   `mnist_subset.py`: Load a subset of the data
-   Graph Structure:
    -   `graph.py`: Interface
    -   `mnist_graph.py`: Load subset of MNIST data as graph
-   Main Algorithms:
    -   `lp.py`: Implement LP
    -   `TSA.py`: Implement TSA (`test.py` implements a toy test case)
    -   `VM.py`: Implement VOpt
    -   `active_SVM.py`, `kmenoids.py`: Implmenet active SVM
-   Testing:
    -   `test_lp_random.py`: LP with random queries
    -   `test_lp_tsa.py`: LP with TSA queries (newer version of `mnist_tsa.py`)
    -   `test_lp_vm.py`: LP with VOpt queries
-   Plotting:
    -   `generate_plots.py`: Plot testing accuracies
