
import os
import socket
from logging import getLogger
import sys 

from accelerate.logging import get_logger
import logging
import time
from accelerate import Accelerator
logger = getLogger('logger')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
def setup_accelerate_logger(accelerator, config):
    """Setup Accelerate's logger with file output for multi-node environments"""
    try:
        # Get node information
        hostname = socket.gethostname()
        rank = accelerator.process_index
        local_rank = accelerator.local_process_index
        
        # Create log directory structure - ABSOLUTE path
        log_dir = os.path.abspath(os.path.join(
            config.output_folders.main_output,
            config.train.model_name,
            config.output_folders.log_dir
        ))
        
        # Only main process creates the directory
        if accelerator.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Log directory created at: {log_dir}")
         
        accelerator.wait_for_everyone()
         
        time.sleep(1)
         
        if accelerator.is_main_process or local_rank == 0: 
            log_file = os.path.join(
                log_dir, 
                f"{'main' if accelerator.is_main_process else hostname}_rank{rank}.log"
            )
             
            root_logger = logging.getLogger()
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(file_handler)
            
            # Get the accelerate logger too
            logger = get_logger(__name__, log_level="INFO")
            
            # Test if logging works
            print(f"Created log file at: {log_file}")
            logger.info(f"Log file created at: {log_file}")
            logger.info(f"Node: {hostname}, Rank: {rank}, Local Rank: {local_rank}")
            return logger
        
        # For other processes, just get the console logger
        return get_logger(__name__, log_level="INFO")
    except Exception as e:
        print(f"Error setting up logger: {e}")
        # Return a basic logger that at least logs to console
        return get_logger(__name__, log_level="INFO")
 

def write_to_main_log(accelerator: Accelerator,result:str, type: str = 'info', check_main_process=True):
    if accelerator.is_main_process or not check_main_process:  
        if type=='warning':
            logger.warning(f"Global Rank: {accelerator.process_index} - {result}")
        if type=='info':
            logger.info(f"Global Rank: {accelerator.process_index} - {result}")
        if type=='error':
            logger.error(f"Global Rank: {accelerator.process_index} - {result}")

def write_to_node_logs(accelerator: Accelerator,result:str, type: str = 'info', check_main_process=True): 
 
    if accelerator.local_process_index==0 or not check_main_process :  
        if type=='warning':
            logger.warning(f"Global Rank: {accelerator.process_index} - {result}")
        if type=='info':
            logger.info(f"Global Rank: {accelerator.process_index} - {result}")
        if type=='error':
            logger.error(f"Global Rank: {accelerator.process_index} - {result}")