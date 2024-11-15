import argparse
import requests
import time
import datetime
import pandas
import signal
import logging

# logging.basicConfig(filename="metrics_recorder.log", filemode="a",
                    # format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    # datefmt='%H:%M:%S',
                    # level=logging.DEBUG)
logger = logging.Logger(__name__)
logger.setLevel(logging.DEBUG)

import multiprocessing as mp
import multiprocessing.synchronize as mpsync

def parse_metrics_response(raw_metrics_text : str, model_name : str):
    # The status_code must be validated before sending the raw text to this function
    metrics = {}
    for line in raw_metrics_text.splitlines():
        if line.startswith("#"):
            continue
        try:
            met_parts = line.split()
            met_name = met_parts[0].replace("vllm:", "").replace(",model_name="+model_name, "").replace("model_name="+model_name, "")
            metrics[met_name] = float(met_parts[1])
        except RuntimeError as e:
            print("Skipping parsing : " + line)

    return metrics

def dump_to_csv(metrics_list, dump_name):
    df = pandas.DataFrame(metrics_list)
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    if dump_name:
        # Dump to csv/jsonl
        df.to_csv(path_or_buf=dump_name)
    else:
        print(df.head())

def record_metrics_func(base_endpoint=None, wait_until_ready=False, kill_signal : mpsync.Event =None, runtime=None, timeout=None, interval = 1, dump_path = "vllm_metrics.csv"):
    if not base_endpoint:
        raise ValueError("Endpoint invalid : " + base_endpoint)
    
    if kill_signal.is_set():
        raise RuntimeError("kill_signal must be unset at the time of starting metrics recording.")
    
    if isinstance(kill_signal, mpsync.Event) and not runtime:
        # Run for as long as kill_signal is not set. 
        runtime = float('inf')

    # Health Check
    health_endpoint = base_endpoint + "/health"
    healthy = None

    if wait_until_ready:
        start_time = time.time()
        while not healthy:
            health_check_resp = requests.get(health_endpoint)

            if (time.time() - start_time) > timeout:
                raise TimeoutError("The vLLM instance is not reachable/is not healthy yet. Timed out")
            
            if health_check_resp.status_code == 200:
                healthy = True
    else:
        if requests.get(health_endpoint).status_code != 200:
            raise RuntimeError("The vLLM instance is not healthy at : " + base_endpoint + ". Exiting")

    # Get model name
    models_resp = requests.get(url = base_endpoint + "/v1/models")
    model_name = models_resp.json()["data"][0]["id"]

    metrics_endpoint = base_endpoint + "/metrics"

    all_metrics = []
    first_rec_time = time.time()
    while time.time() - first_rec_time < runtime :
        if kill_signal.is_set():
            break
        try:
            time.sleep(interval)
            current_timestamp = datetime.datetime.now()
            metrics_resp = requests.get(metrics_endpoint)
            metrics_dict = parse_metrics_response(metrics_resp.text, model_name=model_name)
            metrics_dict["timestamp"] = current_timestamp.isoformat()
            all_metrics.append(metrics_dict)
        except ConnectionError as e:
            # Unable to record - log it
            logging.error("Metrics check failed due to : " + str(e))
        except KeyboardInterrupt as ke:
            logger.info("Keyboard Interrup received. Exiting.")
            
    dump_to_csv(all_metrics, dump_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="localhost")
    parser.add_argument("--wait_until_ready", action="store_true")
    parser.add_argument("--timeout", default=60)
    parser.add_argument("--interval", default=0.2, type=float)
    parser.add_argument("--dump_path", default="metrics.csv")
    parser.add_argument("--runtime_seconds", required=True, type=int)
    args = parser.parse_args()

    kill_sig = mp.Event()

    record_proc = mp.Process(
        target=record_metrics_func,
        args=(args.endpoint, args.wait_until_ready, kill_sig, 600, 60, args.interval, args.dump_path)
    )

    try:
        record_proc.start()
        start_time = time.time()
        while time.time() - start_time < args.runtime :
            time.sleep(args.interval)
    except KeyboardInterrupt as e:
        kill_sig.set()