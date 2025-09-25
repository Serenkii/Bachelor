#!/bin/bash

# Define paths
REMOTE_USER="marian.gunsch"
CLUSTER_HOST="scc"
NODE_HOST="scc066"
PROJECT_DIR="/home/user/marian.gunsch/PycharmProjects/Bachelor/"
SCRIPT="/home/user/marian.gunsch/anaconda3/envs/DataAnalysis/bin/python main_thesis.py PDF"

cd ${PROJECT_DIR}
${SCRIPT}

