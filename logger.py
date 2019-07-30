# adapted from: 
# https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p_logger.py

import logging
import os

def setup_logs(save_dir, run_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = logging.getLogger('Planet-Imagery')
    logger.setLevel(logging.INFO)
 
    # create the logging file handler
    log_file = os.path.join(save_dir, run_name + ".log")
    fh = logging.FileHandler(log_file)
    
    # create the logging console handler
    ch = logging.StreamHandler()
    
    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    
    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 
