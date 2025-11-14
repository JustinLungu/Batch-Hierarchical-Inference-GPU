import time
import json
import requests
import logging

# Function: Send batch to Edge Server
def send_batch_to_server(batch, edge_server_url):
    metadata = [{"UUID": uuid, "Filename": f.filename} for f, uuid in batch]
    data = {"metadata": json.dumps(metadata)}
    files = [("files", (f.filename, f.file, f.content_type)) for f, _ in batch]

    dynamic_timeout = max(10, len(batch) * 60)

    try:

        # ts_sample_sent_to_edge_server: Request sent to Edge Server time
        ts_sample_sent_to_edge_server = time.time()

        response = requests.post(edge_server_url, files=files, data=data, timeout=dynamic_timeout)
        if response.status_code == 200:

            # ts_results_received_from_edge_server: Response received from Edge Server time
            ts_results_received_from_edge_server = time.time()

            return (
                response.json(),
                ts_sample_sent_to_edge_server,
                ts_results_received_from_edge_server,
            )
        logging.error(f"Batch offload failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Batch offload exception: {e}")
    return None

# Function: Flush dynamic batching buffer
def flush_dynamic_buffer(edge_server_url):
    if hasattr(offload_sample_method, "dynamic_buffer") and offload_sample_method.dynamic_buffer:
        batch = offload_sample_method.dynamic_buffer
        offload_sample_method.dynamic_buffer = []
        return send_batch_to_server(batch, edge_server_url)
    return None

# ---- Main Function ----
def offload_sample_method(file, offloading_strategy, edge_server_url, batch_size, batch_wait_time):
    def send_batch(batch):
        return send_batch_to_server(batch, edge_server_url)

    # ---- Send individually ----
    if offloading_strategy == "send_individually":
        logging.info(f"Sample: {file.filename}, Offload strategy: Send individually")
        file.file.seek(0)
        metadata = [{"UUID": file.uuid, "Filename": file.filename}]
        data = {"metadata": json.dumps(metadata)}
        files = [("files", (file.filename, file.file, file.content_type))]

        try:
            # ts_sample_sent_to_edge_server: Request sent to Edge Server time
            ts_sample_sent_to_edge_server = time.time()

            response = requests.post(edge_server_url, files=files, data=data, timeout=10)
            if response.status_code == 200:

                # ts_results_received_from_edge_server: Response received from Edge Server time
                ts_results_received_from_edge_server = time.time()

                return (response.json(), ts_sample_sent_to_edge_server, ts_results_received_from_edge_server)
            else:
                logging.error(f"Offload failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Request exception: {e}")
            return None

    # ---- Size-based batching ----
    elif offloading_strategy == "size_based_batching":
        logging.info(f"Sample: {file.filename}, Offload strategy: Size-based batching")
        if not hasattr(offload_sample_method, "batch_buffer_size"):
            offload_sample_method.batch_buffer_size = []

        file.file.seek(0)
        offload_sample_method.batch_buffer_size.append((file, file.uuid))
        logging.info(f"Sample: {file.filename}, Added to buffer, Current size: {len(offload_sample_method.batch_buffer_size)}")

        if len(offload_sample_method.batch_buffer_size) >= batch_size:
            batch = offload_sample_method.batch_buffer_size
            offload_sample_method.batch_buffer_size = []
            return send_batch(batch)

        return None

    # ---- Dynamic batching ----
    elif offloading_strategy == "dynamic_batching":
        logging.info(
            f"Sample: {file.filename}, Offload strategy: Dynamic batching"
        )
        if not hasattr(offload_sample_method, "dynamic_buffer"):
            offload_sample_method.dynamic_buffer = []
        file.file.seek(0)
        offload_sample_method.dynamic_buffer.append((file, file.uuid))

        return None

    # ---- Unknown strategy ----
    logging.error(f"Invalid offload strategy: {offloading_strategy}")
    return None
