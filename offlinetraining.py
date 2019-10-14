import os
import sys

# Copypasta and modification of function for offline Training on remote machine with pycharm.
# Original source: Hooman Shayani (https://medium.com/@hooshaya/train-your-model-in-background-and-disconnect-from-pycharm-remote-debug-run-over-ssh-with-nohup-563cb778a1e9)

os.system("nohup sh -c '" +
          sys.executable + " train.py" +
          "' &")

# Close laptop, go home
