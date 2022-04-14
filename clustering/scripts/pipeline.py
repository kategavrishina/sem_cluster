import os

os.system(os.path.join(os.pardir,'data2join.py') + " _ " + os.path.join(os.pardir, 'sem_cluster','день','uses.csv') + os.path.join(os.pardir, 'sem_cluster','день','judgments.csv') + " None True joined_file.csv")
