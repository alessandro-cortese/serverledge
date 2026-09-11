"""
Locustfile dedicato alla raccolta controllata dei campioni di profiling.

Obiettivi:
- eseguire soltanto le funzioni del batch indicato;
- avere esattamente un Locust user per funzione;
- effettuare un numero finito di richieste per funzione;
- mantenere concorrenza = 1 per ogni singola funzione;
- salvare nel CSV i campi utili per distinguere cold/warm e validità
  del ResourceProfile.

Variabili d'ambiente:

FUNCTIONS_TO_RUN
    Lista separata da virgole dei nomi Serverledge da invocare.
    Esempio:
    hashing,hashing-py,json-dumps-node

REQUESTS_PER_FUNCTION
    Numero di richieste Locust per funzione.
    Default: 16.

LB_POLICY
    Policy del load balancer, salvata nel CSV.

COLLECTION_CSV
    Nome del CSV di output.
    Default: experiment_results.csv

Il numero di utenti Locust deve coincidere con il numero delle funzioni
contenute in FUNCTIONS_TO_RUN.
"""

import csv
import os
import re
import time

import gevent
from gevent.lock import Semaphore
from locust import HttpUser, constant, events, task
from locust.exception import StopUser


CSV_FILE = os.environ.get("COLLECTION_CSV", "experiment_results.csv")
REQUESTS_PER_FUNCTION = int(os.environ.get("REQUESTS_PER_FUNCTION", "16"))

_raw_functions = os.environ.get("FUNCTIONS_TO_RUN", "")
FUNCTIONS_TO_RUN = [
    function_name.strip()
    for function_name in _raw_functions.split(",")
    if function_name.strip()
]

if not FUNCTIONS_TO_RUN:
    raise RuntimeError(
        "FUNCTIONS_TO_RUN non impostata. "
        "Esempio: FUNCTIONS_TO_RUN=hashing,hashing-py"
    )

if REQUESTS_PER_FUNCTION <= 0:
    raise RuntimeError("REQUESTS_PER_FUNCTION deve essere > 0")

if len(set(FUNCTIONS_TO_RUN)) != len(FUNCTIONS_TO_RUN):
    raise RuntimeError("FUNCTIONS_TO_RUN contiene nomi duplicati")


# Timeout più elevati per i benchmark notoriamente più lenti.
FUNCTION_TIMEOUTS = {
    "primenumber": 300,
    "readmemory": 120,
    "thread": 320,
    "linpack": 120,
    "filehandle": 420,
    "randomaccess-py": 120,
    "pointerchase-py": 90,
}

DEFAULT_TIMEOUT = 60


_csv_lock = Semaphore()
_completion_lock = Semaphore()
_completed_functions = set()


CSV_HEADER = [
    "timestamp",
    "function",
    "sample_index",
    "response_time_s",
    "duration_s",
    "init_time_s",
    "is_warm_start",
    "execution_succeeded",
    "node_arch",
    "status_code",
    "policy",
    "locust_response_time_ms",
    "profile_valid",
    "exclusive_container",
]


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    with _csv_lock:
        with open(CSV_FILE, "w", newline="") as csv_file:
            csv.writer(csv_file).writerow(CSV_HEADER)

    print(f"[COLLECTION] functions={','.join(FUNCTIONS_TO_RUN)}")
    print(f"[COLLECTION] requests_per_function={REQUESTS_PER_FUNCTION}")


@events.request.add_listener
def on_request(
        request_type,
        name,
        response_time,
        response_length,
        response,
        exception,
        context,
        **kwargs,
):
    policy = os.environ.get("LB_POLICY", "unknown")
    sample_index = context.get("sample_index", "") if context else ""

    serverledge_response_time = ""
    duration = ""
    init_time = ""
    warm_start = ""
    execution_succeeded = ""
    node_arch = "unknown"
    status_code = ""
    profile_valid = ""
    exclusive_container = ""

    if response is not None:
        status_code = response.status_code
        node_arch = response.headers.get("Serverledge-Node-Arch", "unknown")

        try:
            data = response.json()

            serverledge_response_time = data.get("ResponseTime", "")
            duration = data.get("Duration", "")
            init_time = data.get("InitTime", "")
            warm_start = data.get("IsWarmStart", "")
            execution_succeeded = data.get("Success", "")

            profile = data.get("ResourceProfile") or {}
            profile_valid = profile.get("valid", "")
            exclusive_container = profile.get("exclusive_container", "")
        except Exception:
            pass

    if exception is not None:
        status_code = f"FAILED:{type(exception).__name__}"

    row = [
        time.time(),
        name,
        sample_index,
        serverledge_response_time,
        duration,
        init_time,
        warm_start,
        execution_succeeded,
        node_arch,
        status_code,
        policy,
        response_time or 0,
        profile_valid,
        exclusive_container,
        ]

    with _csv_lock:
        with open(CSV_FILE, "a", newline="") as csv_file:
            csv.writer(csv_file).writerow(row)


def mark_function_completed(function_name, environment):
    should_quit = False

    with _completion_lock:
        _completed_functions.add(function_name)

        print(
            f"[COLLECTION] completed {function_name} "
            f"({len(_completed_functions)}/{len(FUNCTIONS_TO_RUN)})"
        )

        if len(_completed_functions) == len(FUNCTIONS_TO_RUN):
            should_quit = True

    if should_quit:
        # Consente alla task corrente di terminare prima di fermare Locust.
        gevent.spawn_later(0.5, environment.runner.quit)


def make_user_class(function_name):
    timeout = FUNCTION_TIMEOUTS.get(function_name, DEFAULT_TIMEOUT)

    class CollectionUser(HttpUser):
        wait_time = constant(0.0)
        fixed_count = 1

        function = function_name
        function_timeout = timeout

        def on_start(self):
            self.sample_index = 0

        @task
        def invoke(self):
            if self.sample_index >= REQUESTS_PER_FUNCTION:
                raise StopUser()

            self.sample_index += 1

            with self.client.post(
                    f"/invoke/{self.function}",
                    json={"params": {}},
                    name=self.function,
                    timeout=self.function_timeout,
                    context={"sample_index": self.sample_index},
                    catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"HTTP {response.status_code}")
                else:
                    try:
                        data = response.json()
                    except Exception as exc:
                        response.failure(f"Invalid JSON: {exc}")
                    else:
                        if not data.get("Success", False):
                            response.failure("Serverledge Success=false")

            if self.sample_index >= REQUESTS_PER_FUNCTION:
                mark_function_completed(self.function, self.environment)
                raise StopUser()

    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", function_name)
    CollectionUser.__name__ = f"Collect_{safe_name}"
    CollectionUser.__qualname__ = CollectionUser.__name__

    return CollectionUser


# Locust scopre automaticamente le User class presenti nel namespace
# del modulo. Ne creiamo una per ogni funzione richiesta.
for _function_name in FUNCTIONS_TO_RUN:
    _user_class = make_user_class(_function_name)
    globals()[_user_class.__name__] = _user_class

del _function_name
del _user_class