import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

def run_with_timeout(timeout, func, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        result = None
    finally:
        signal.alarm(0)
    return result

# import concurrent.futures

# class TimeoutException(Exception):
#     pass

# def run_with_timeout(timeout, func, *args, **kwargs):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(func, *args, **kwargs)
#         try:
#             result = future.result(timeout=timeout)
#         except concurrent.futures.TimeoutError:
#             raise TimeoutException
#     return result