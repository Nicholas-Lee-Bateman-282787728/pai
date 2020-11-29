import multiprocessing
import os
from logic.baseline import Baseline
# W: Word, S:syllable, P: Frequency_Phoneme, SP: One_Phonemes, AP: All_Phonemes
list_model = ['11111', '01111', '10111', '00111', '11011', '01011', '10011', '00011',
              '11101', '01101', '10101', '00101', '11001', '01001', '10001', '00001',
              '11110', '01110', '10110', '00110', '11010', '01010', '10010', '00010',
              '11100', '01100', '10100', '00100', '11000', '01000', '10000', '00000']

bl_es = Baseline()
report_name = 'SentimentWSP_Affect'
# printing main program process id
print("ID of main process: {}".format(os.getpid()))
# creating processes

jobs = []
for i in list_model:
    p = multiprocessing.Process(target=bl_es.main, args=('es', report_name, list_model[0], True, [-1, 0, 1]))
    jobs.append(p)
    p.start()
    print("ID of process p: {}".format(p.pid))


# wait until processes are finished
[p.join() for p in jobs]

# check if processes are alive
for p in jobs:
    print("Process p={0} is alive: {1}".format(p.pid, p.is_alive()))

