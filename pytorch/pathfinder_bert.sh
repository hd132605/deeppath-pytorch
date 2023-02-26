#!/bin/bash

relation=$1
python sl_policy_bert.py $relation
python policy_agent_bert.py $relation retrain
python policy_agent_bert.py $relation test