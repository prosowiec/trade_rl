import psutil
import os
import subprocess
import sys
import time
import logging



def find_process(script_name):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if (
                proc.info['name']
                and proc.info['name'].startswith("python")
                and len(proc.info['cmdline']) > 1
                and script_name in proc.info['cmdline'][1]
                and proc.pid != os.getpid()
            ):
                return proc
        except (TypeError):
            continue
    return None


def start_main(script_name):
    subprocess.Popen([sys.executable, script_name, "--dashboard"])
    time.sleep(1)


def stop_main(proc):
    """Zatrzymuje proces main.py."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except psutil.TimeoutExpired:
        proc.kill()

def restart(script_name):
    proc = find_process(script_name)
    if proc:
        stop_main(proc)
    
    start_main(script_name)