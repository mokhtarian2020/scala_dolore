#!/bin/bash
cd /home/amir/Documents/amir/CATO\ MAIOR/project
source venv/bin/activate
export PYTHONPATH="/home/amir/Documents/amir/CATO MAIOR/project:$PYTHONPATH"
cd backend
python main.py
