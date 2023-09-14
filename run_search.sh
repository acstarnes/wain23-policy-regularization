#!/bin/sh

python3 -m main -c mnist
python3 -m main -c cifar10
python3 -m main -c spotify

#python3 -m main -c mnist_search_mmd
#python3 -m main -c mnist_search_ent
#python3 -m main -c mnist_search_l1
#python3 -m main -c mnist_search_l2
#python3 -m main -c cifar10_search_mmd
#python3 -m main -c cifar10_search_ent
#python3 -m main -c cifar10_search_l1
#python3 -m main -c cifar10_search_l2
#python3 -m main -c spotify_search_mmd
#python3 -m main -c spotify_search_ent
#python3 -m main -c spotify_search_l1
#python3 -m main -c spotify_search_l2

