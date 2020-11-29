import multiprocessing
import os
from logic.baseline import Baseline

bl = Baseline()
report_name = 'SentimentWSP_Affect'

# printing main program process id
print("ID of main process: {}".format(os.getpid()))
# creating processes

jobs = []

p = multiprocessing.Process(target=bl.main, args=('en', report_name, '11110', True, [-1, 0, 1]))
p.start()
print("ID of process p: {}".format(p.pid))


# wait until processes are finished
p.join()

# check if processes are alive

print("Process p{0}={1} is alive: {2}".format(p, p.pid, p.is_alive()))

