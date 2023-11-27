import csv
import platform
import time
from datetime import datetime
from multiprocessing import Process, Value
from os import makedirs, path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

if platform.system() != "Windows":
    try:
        from nvitop import Device
    except Exception:
        Device = None

# Constants
BYTES_PER_GB = 1024 * 1024 * 1024


def bytes_to_gb(bytes_value: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_value / BYTES_PER_GB


def log_device_info(device, prefix: str, csvwriter, display_log):
    """
    Logs device information.

    Parameters:
        device: The device object with GPU information.
        prefix: The prefix string for log identification.
        csvwriter: The CSV writer object for writing logs to a file.
        display_log: print out on shell
    """
    total_memory_gb = float(device.memory_total_human()[:-3])
    used_memory_bytes = device.memory_used()
    gpu_utilization = device.gpu_utilization()

    if display_log:
        print(f"Device: {device.name}")
        print(f"  - Used memory    : {bytes_to_gb(used_memory_bytes):.2f} GB")
        print(f"  - Used memory%   : {bytes_to_gb(used_memory_bytes)/total_memory_gb * 100:.2f}%")
        print(f"  - GPU utilization: {gpu_utilization}%")
        print("-" * 40)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (bytes_to_gb(used_memory_bytes) / total_memory_gb) * 100

    csvwriter.writerow([current_time, bytes_to_gb(used_memory_bytes), gpu_utilization])


def monitor_and_plot(prefix="result/tmp", display_log=False, stop_flag: Value = None):
    """Monitor and plot GPU usage.
    Args:
        prefix: The prefix of the output file.
        stop_flag: A multiprocessing.Value to indicate if monitoring should stop.
    """
    devices = Device.all()
    initial_pids = set()
    monitored_pids = set()

    with open(f"{prefix}.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Time", "Used Memory%", "GPU Utilization"])

        try:
            while True:
                if stop_flag and stop_flag.value:
                    break

                for device in devices:
                    current_pids = set(device.processes().keys())
                    if not initial_pids:
                        initial_pids = current_pids

                    new_pids = current_pids - initial_pids
                    if new_pids:
                        monitored_pids.update(new_pids)

                    for pid in monitored_pids.copy():
                        if pid not in current_pids:
                            monitored_pids.remove(pid)
                            if not monitored_pids:
                                raise StopIteration
                    log_device_info(device, prefix, csvwriter, display_log)
                time.sleep(1)
        except StopIteration:
            pass

    plot_data(prefix)
    return


def plot_data(prefix):
    """Plot the data from the CSV file.
    Args:
        prefix: The prefix of the CSV file.
    """
    data = list(csv.reader(open(f"{prefix}.csv")))
    if len(data) < 2:
        print("Insufficient data for plotting.")
        return

    time_stamps, used_memory, gpu_utilization = zip(*data[1:])
    used_memory = [float(x) for x in used_memory]
    gpu_utilization = [float(x) for x in gpu_utilization]
    if len(used_memory) >= 10:
        tick_spacing = len(used_memory) // 10
    else:
        tick_spacing = 1

    try:
        plot_graph(
            time_stamps,
            used_memory,
            "Used Memory (GB)",
            "Time",
            "Used Memory (GB)",
            "Used Memory Over Time",
            tick_spacing,
            f"{prefix}_memory.png",
        )
    except Exception as e:
        message = f"plot_graph of Memory error, error info:{str(e)}"
        print(message)

    try:
        plot_graph(
            time_stamps,
            gpu_utilization,
            "GPU Utilization (%)",
            "Time",
            "GPU Utilization (%)",
            "GPU Utilization Over Time",
            tick_spacing,
            f"{prefix}_utilization.png",
        )
    except Exception as e:
        message = f"plot_graph of Utilization error, error info:{str(e)}"
        print(message)


def plot_graph(x, y, label, xlabel, ylabel, title, tick_spacing, filename):
    """Generate and save a plot.
    Args:
        x: X-axis data.
        y: Y-axis data.
        label: The label for the plot.
        xlabel: Label for X-axis.
        ylabel: Label for Y-axis.
        title: The title of the plot.
        tick_spacing: Interval for tick marks on the x-axis.
        filename: The filename to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(rotation=45)
    plt.savefig(filename)


def gpu_monitor_decorator(prefix="result/gpu_info", display_log=False):
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            if platform.system() != "Windows" and Device is not None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                dynamic_prefix = f"{prefix}/{func.__name__}_{timestamp}"

                directory = path.dirname(dynamic_prefix)
                if not path.exists(directory):
                    try:
                        makedirs(directory)
                    except Exception as e:
                        comment = f"GPU Info record need a result/gpu_info dir in your SDWebUI, now failed with {str(e)}"
                        print(comment)
                        dynamic_prefix = f"{func.__name__}_{timestamp}"

                stop_flag = Value("b", False)

                monitor_proc = Process(target=monitor_and_plot, args=(dynamic_prefix, display_log, stop_flag))
                monitor_proc.start()

                try:
                    result = func(*args, **kwargs)
                finally:
                    stop_flag.value = True
                    monitor_proc.join()
            else:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return actual_decorator


if __name__ == "__main__":
    pass

    # Display how to define a GPU infer function and wrap with gpu_monitor_decorator
    @gpu_monitor_decorator()
    def execute_process(repeat=5):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        retina_face_detection = pipeline(Tasks.face_detection, "damo/cv_resnet50_face-detection_retinaface")
        img_path = "https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/retina_face_detection.jpg"

        for i in range(repeat):
            retina_face_detection([img_path] * 10)
        return

    if 1:
        execute_process(repeat=5)
