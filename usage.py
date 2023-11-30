import time
import gpustat, subprocess, GPUtil
import threading
import psutil


class TrackParams(object):
    def __init__(self, stats_func):
        self.cpu_percent_sum = 0.0
        self.cpu_percent_count = 0
        self.cpu_memory_max = float('-inf')
        self.cpu_memory_sum = 0.0
        self.cpu_memory_count = 0
        self.gpu_percent_sum = 0.0
        self.gpu_percent_count = 0
        self.gpu_memory_max = float('-inf')
        self.gpu_memory_sum = 0.0
        self.gpu_memory_count = 0
        self.time_start = None
        self.thread_event = threading.Event()
        self.thread = threading.Thread(target=stats_func)
        self.mb_conv = 1024 ** 2


class TrackUsage(object):
    def __init__(self):
        self.params = TrackParams(self.__track_usage)
        self.avg_cpu_perc = None
        self.avg_cpu_memory = None
        self.max_cpu_memory = None
        self.cpu_memory_start = None
        self.avg_gpu_perc = None
        self.avg_gpu_memory = None
        self.max_gpu_memory = None
        self.gpu_memory_start = None
        self.time_duration = None

    def start(self):
        self.__reset_track(keep_results=False)
        self.cpu_memory_start = psutil.cpu_percent()
        self.gpu_memory_start = [gpu.memory_used for gpu in gpustat.new_query().gpus]
        self.gpu_memory_start = sum(self.gpu_memory_start) / len(self.gpu_memory_start)

        self.params.time_start = time.time()
        self.params.thread.start()

    def stop(self):
        self.params.thread_event.set()
        self.params.thread.join()
        self.__process_results()
        self.__reset_track()

    def __process_results(self):
        self.time_duration = time.time() - self.params.time_start
        self.avg_cpu_perc = self.params.cpu_percent_sum / self.params.cpu_percent_count\
            if self.params.cpu_percent_count > 0 else 0
        self.avg_cpu_memory = self.params.cpu_memory_sum / self.params.cpu_memory_count\
            if self.params.cpu_memory_count > 0 else 0
        self.max_cpu_memory = self.params.cpu_memory_max if self.params.cpu_memory_max != float('-inf') else 0
        self.avg_gpu_perc = self.params.gpu_percent_sum / self.params.gpu_percent_count\
            if self.params.gpu_percent_count > 0 else 0
        self.avg_gpu_memory = self.params.gpu_memory_sum / self.params.gpu_memory_count\
            if self.params.gpu_memory_count > 0 else 0
        self.max_gpu_memory = self.params.gpu_memory_max if self.params.gpu_memory_max != float('-inf') else 0

    def __reset_track(self, keep_results=True):
        self.params = TrackParams(self.__track_usage)

        if not keep_results:
            self.avg_cpu_perc = None
            self.avg_cpu_memory = None
            self.max_cpu_memory = None
            self.avg_gpu_perc = None
            self.avg_gpu_memory = None
            self.max_gpu_memory = None
            self.time_duration = None

    def __track_usage(self):
        while not self.params.thread_event.is_set():
            self.params.cpu_percent_sum += psutil.cpu_percent()
            self.params.cpu_percent_count += 1
            cpu_memory = psutil.virtual_memory().used / self.params.mb_conv
            self.params.cpu_memory_sum += cpu_memory
            self.params.cpu_memory_count += 1
            self.params.cpu_memory_max = max(self.params.cpu_memory_max, cpu_memory)
            gpu_percents = [perc for perc in self.__get_gpu_perc()]
            self.params.gpu_percent_sum += sum(gpu_percents)
            self.params.gpu_percent_count += len(gpu_percents)
            gpu_memory = [gpu.memory_used for gpu in gpustat.new_query().gpus]
            self.params.gpu_memory_sum += sum(gpu_memory)
            self.params.gpu_memory_count += len(gpu_memory)
            self.params.gpu_memory_max = max([self.params.gpu_memory_max] + gpu_memory)

            time.sleep(1)

    @staticmethod
    def __get_gpu_perc():
        try:
            gpu_cores = GPUtil.getGPUs()
            return [gpu.load * 100 for gpu in gpu_cores]
        except Exception as e:
            print(f'Error retrieving GPU Percent {e}')
            return [0.0 for _ in range(len(gpustat.new_query().gpus))]
