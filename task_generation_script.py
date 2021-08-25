import os
import shutil
import json
from envs import PROJECT_FOLDER

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def generate_animal_zeroshot_bash():
    relative_path = PROJECT_FOLDER + '/'
    job_path = 'toy_example_jobs/'
    #================================================
    jobs_path = relative_path + job_path
    os.makedirs(jobs_path, exist_ok=True)
    # ================================================
    if os.path.exists(jobs_path):
        remove_all_files(jobs_path)
    ##################################################
    with open(jobs_path + 'toy_example_zeroshot_animal.sh', 'w') as rsh_i:
        command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.py --sent_dropout 0.1"
        rsh_i.write(command_i)
        print('saving jobs at {}'.format(jobs_path + 'toy_example_animal_sentdrop0.1.sh'))

    with open(jobs_path + 'toy_example_zeroshot_animal_sentdrop0.1.sh', 'w') as rsh_i:
        command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.py --sent_dropout 0.1"
        rsh_i.write(command_i)
        print('saving jobs at {}'.format(jobs_path + 'toy_example_animal_sentdrop0.1.sh'))

    with open(jobs_path + 'toy_example__zeroshot_animal_sentdrop_beta0.1.sh', 'w') as rsh_i:
        command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m run.py --sent_dropout 0.1 --beta_drop 'true'"
        rsh_i.write(command_i)
        print('saving jobs at {}'.format(jobs_path + 'toy_example_animal_sentdrop0.1.sh'))

if __name__ == '__main__':
    generate_animal_zeroshot_bash()