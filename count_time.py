import time


def execute_time(func):
    def int_time(*args, **kwargs):
        print(">>>>开始运行" + func.__name__ + "程序")
        start_time = time.time()  # 程序开始时间
        result = func(*args)
        over_time = time.time()  # 程序结束时间
        total_time = over_time - start_time
        print(">>>>程序{}共计{:.3f}秒".format(func.__name__, total_time))
        return result
    return int_time
