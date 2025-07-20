import psutil
import os
import gc

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(stage=""):
    memory_mb = get_memory_usage()
    print(f"Memory usage {stage}: {memory_mb:.2f} MB")
    return memory_mb

def optimize_memory():
    gc.collect()
    memory_mb = get_memory_usage()
    print(f"Memory after optimization: {memory_mb:.2f} MB")
    return memory_mb

def check_memory_limit(limit_mb=500):
    memory_mb = get_memory_usage()
    if memory_mb > limit_mb:
        print(f"⚠️  Memory usage ({memory_mb:.2f} MB) exceeds limit ({limit_mb} MB)")
        return False
    return True 