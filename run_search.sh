#!/bin/sh

python3 -m main -c mnist_search_pg_ent
python3 -m main -c mnist_search_pg_l1
python3 -m main -c mnist_search_pg_l2
python3 -m main -c mnist_search_pg_mmd
python3 -m main -c mnist_search_q_ent
python3 -m main -c mnist_search_q_l1
python3 -m main -c mnist_search_q_l2
python3 -m main -c mnist_search_q_mmd
python3 -m main -c cifar10_search_pg_ent
python3 -m main -c cifar10_search_pg_l1
python3 -m main -c cifar10_search_pg_l2
python3 -m main -c cifar10_search_pg_mmd
python3 -m main -c cifar10_search_q_ent
python3 -m main -c cifar10_search_q_l1
python3 -m main -c cifar10_search_q_l2
python3 -m main -c cifar10_search_q_mmd
python3 -m main -c cifar100_search_pg_ent
python3 -m main -c cifar100_search_pg_l1
python3 -m main -c cifar100_search_pg_l2
python3 -m main -c cifar100_search_pg_mmd
python3 -m main -c cifar100_search_q_ent
python3 -m main -c cifar100_search_q_l1
python3 -m main -c cifar100_search_q_l2
python3 -m main -c cifar100_search_q_mmd

