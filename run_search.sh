#!/bin/sh

#python3 -m main -c mnist
#python3 -m main -c cifar10
#python3 -m main -c spotify

python3 -m main -c mnist_search_mmd
python3 -m main -c mnist_search_ent
python3 -m main -c mnist_search_js
python3 -m main -c mnist_search_hl
python3 -m main -c mnist_search_tv

python3 -m main -c spotify_search_mmd
python3 -m main -c spotify_search_ent
python3 -m main -c spotify_search_js
python3 -m main -c spotify_search_hl
python3 -m main -c spotify_search_tv

python3 -m main -c cifar10_search_mmd
python3 -m main -c cifar10_search_ent
python3 -m main -c cifar10_search_js
python3 -m main -c cifar10_search_hl
python3 -m main -c cifar10_search_tv

