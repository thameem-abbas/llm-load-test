"""Main llm-load-test CLI entrypoint."""

import logging
import logging.handlers
import multiprocessing as mp
import sys
import time
from aux_utils.vllm_recorder import record_metrics_func
from user import User

from dataset import Dataset

import logging_utils

import utils


def run_main_process(concurrency, duration, dataset, dataset_q, stop_q, batch_size=1):
    """Run the main process."""
    logging.info("Test from main process")

    # Initialize the dataset_queue with 4*concurrency requests
    for query in dataset.get_next_n_queries(2 * concurrency * batch_size):
        dataset_q.put(query)

    start_time = time.time()
    current_time = start_time
    while (current_time - start_time) < duration:
        # Keep the dataset queue full for duration
        if dataset_q.qsize() < int(0.5 * concurrency * batch_size + 1):
            logging.info("Adding %d entries to dataset queue", concurrency * batch_size)
            for query in dataset.get_next_n_queries(concurrency * batch_size):
                dataset_q.put(query)
        time.sleep(0.1)
        current_time = time.time()

    logging.info("Timer ended, stopping processes")

    # Signal users to stop sending requests
    stop_q.put(None)

    # Empty the dataset queue
    while not dataset_q.empty():
        logging.debug("Removing element from dataset_q")
        dataset_q.get()

    return


def gather_results(results_pipes):
    """Get the results."""
    # Receive all results from each processes results_pipe
    logging.debug("Receiving results from user processes")
    results_list = []
    for results_pipe in results_pipes:
        user_results = results_pipe.recv()
        results_list.extend(user_results)
    return results_list


def exit_gracefully(procs, dataset_q, stop_q, logger_q, log_reader_thread, code):
    """Exit gracefully."""
    # Signal users to stop sending requests
    if stop_q.empty():
        stop_q.put(None)

    if dataset_q is not None and not dataset_q.empty():
        logging.warning("Removing more elements from dataset_q after gathering results!")
        while not dataset_q.empty():
            dataset_q.get()

    logging.debug("Calling join() on all user processes")
    for proc in procs:
        proc.join()
    logging.info("User processes terminated succesfully")

    # Shutdown logger thread
    logger_q.put(None)
    log_reader_thread.join()

    sys.exit(code)


def main(args):
    """Load test CLI entrypoint."""
    args = utils.parse_args(args)

    mp_ctx = mp.get_context("spawn")
    logger_q = mp_ctx.Queue()
    log_reader_thread = logging_utils.init_logging(args.log_level, logger_q)

    # Create processes and their Users
    stop_q = mp_ctx.Queue(1)
    dataset_q = mp_ctx.Queue()
    procs = []
    results_pipes = []

    # Metrics Logging
    metrics_conf = config.get("metrics_recorder")
    metrics_save_dir = config["output"]["dir"] + "/metrics"
    metrics_save_name = metrics_save_dir + "/" + config["output"]["file"][:-5] + "_metrics.json"
    kill_sig = mp.Event()
    record_proc = mp.Process(
        target=record_metrics_func,
        args=(config["plugin_options"]["endpoint"], True, kill_sig, None, metrics_conf["timeout"], metrics_conf["interval"], metrics_save_name)
    )

    record_proc.start()

    # Parse config
    logging.debug("Parsing YAML config file %s", args.config)
    concurrency, duration, plugin = 0, 0, None
    try:
        config = utils.yaml_load(args.config)
        print("Config : ", config)
        concurrency, duration, plugin, batch_size = utils.parse_config(config)
    except Exception as e:
        logging.error("Exiting due to invalid input: %s", repr(e))
        kill_sig.set()
        exit_gracefully(procs, dataset_q, stop_q, logger_q, log_reader_thread, 1)
    try:
        logging.debug("Creating dataset with configuration %s", config["dataset"])
        # Get model_name if set for prompt formatting
        model_name = config.get("plugin_options", {}).get("model_name", "")
        dataset = Dataset(model_name=model_name, **config["dataset"])

        logging.debug("Creating %s Users and corresponding processes", concurrency)
        for idx in range(concurrency):
            send_results, recv_results = mp_ctx.Pipe()
            user = User(
                idx,
                dataset_q=dataset_q,
                stop_q=stop_q,
                results_pipe=send_results,
                plugin=plugin,
                logger_q=logger_q,
                log_level=args.log_level,
                run_duration=duration,
                batch_size=batch_size,
            )
            proc = mp_ctx.Process(target=user.run_user_process)
            procs.append(proc)
            logging.info("Starting %s", proc)
            results_pipes.append(recv_results)

        # Attempt to start them closer to each other. With high concurrency numbers,
        # it currently takes up to 10 seconds between the launch of the first and 
        # the last user proc.
        for proc in procs:
            proc.start()

        logging.debug("Running main process")
        run_main_process(concurrency, duration, dataset, dataset_q, stop_q, batch_size)

        results_list = gather_results(results_pipes)

        utils.write_output(config, results_list)

    # Terminate queues immediately on ^C
    except KeyboardInterrupt:
        stop_q.cancel_join_thread()
        dataset_q.cancel_join_thread()
        kill_sig.set()
        exit_gracefully(procs, dataset_q, stop_q, logger_q, log_reader_thread, 130)
    except Exception:
        logging.exception("Unexpected exception in main process")
        kill_sig.set()
        exit_gracefully(procs, dataset_q, stop_q, logger_q, log_reader_thread, 1)

    kill_sig.set()
    exit_gracefully(procs, dataset_q, stop_q, logger_q, log_reader_thread, 0)

if __name__ == "__main__":
    main(sys.argv[1:])
