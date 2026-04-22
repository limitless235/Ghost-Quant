#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
python3 eval/perplexity.py
python3 eval/pareto_curve.py
python3 eval/ilp_vs_greedy.py
