from itertools import cycle
import threading
import time
import sys

def progress_spinner(stop_event):
    spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while not stop_event.is_set():
        sys.stdout.write(f"\rProcessing... {next(spinner)}")
        sys.stdout.flush()
        time.sleep(0.1)

def with_progress_indicator(func):
    def wrapper(*args, **kwargs):
        stop_spinner = threading.Event()
        spinner_thread = threading.Thread(target=progress_spinner, args=(stop_spinner,))
        
        try:
            spinner_thread.start()
            result = func(*args, **kwargs)
            stop_spinner.set()
            spinner_thread.join()
            print("\rComplete!            ")  # Clear the spinner
            return result
        except Exception as e:
            stop_spinner.set()
            spinner_thread.join()
            print("\rError occurred!      ")  # Clear the spinner
            raise e
    
    return wrapper