"""
    The main driver script
"""

from __future__ import print_function
import sys
from mnist import evaluate_mnist
from cifar import evaluate_cifar


if len(sys.argv) != 2:
    print ("The correct syntax for running the script is python3 main.py mnist/cifar ")

elif sys.argv[1] == 'mnist':
    evaluate_mnist()

elif sys.argv[1] == 'cifar':
    evaluate_cifar()

else:
    print ("The correct syntax for running the script is python3 main.py mnist/cifar ")