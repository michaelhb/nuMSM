#!/bin/bash
# Inputs: db file, output csv
sqlite3 -header -separator "," $1 "select rew,delta,eta,bau from points;" > $2