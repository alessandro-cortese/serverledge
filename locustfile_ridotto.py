"""Locustfile ridotto per la validazione della catena sperimentale.

Contiene solo le quattro funzioni che al momento eseguono correttamente sul
cluster GCP: primenum, readdisk, readmemory e thread. Le altre — chacha20,
linpack, filehandle, amd_faster, arm_faster — restano escluse in attesa dei
chiarimenti di Filippo su nomi, parametri e sorgenti.

NON è la replica fedele dell'esperimento originale e i suoi risultati non vanno
confrontati direttamente con quelli della sua tesi: il sottoinsieme di funzioni
è diverso, quindi il carico complessivo e la sua composizione non coincidono.

Serve a verificare end-to-end che locust giri, che il CSV si popoli con
l'architettura di esecuzione e che la raccolta dei risultati funzioni — e a
ottenere un primo confronto RoundRobin contro LinUCB su hardware eterogeneo.

La struttura degli event listener e il formato del CSV sono identici
all'originale di Filippo, così che gli script di analisi restino compatibili.
"""

import csv
import os
import time

from locust import HttpUser, constant, events, task

CSV_FILE = "experiment_results.csv"


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "function",
                    "response_time_s",
                    "node_arch",
                    "status_code",
                    "policy",
                    "locust_response_time",
                ]
            )


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

    if exception:
        print(f"Request failed: {exception}")
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.time(),
                    name,
                    "unknown",
                    "unknown",
                    f"FAILED: {type(exception).__name__}",
                    policy,
                    response_time or 0,
                ]
            )
        return

    # L'architettura che ha eseguito arriva in questo header, impostato dal nodo
    # in api.go e propagato dal load balancer. È il dato che distingue le due
    # politiche: con l'hash ring resta costante per funzione, con il MAB varia.
    node_arch = response.headers.get("Serverledge-Node-Arch", "unknown")

    serverledge_response_time = "unknown"

    try:
        data = response.json()
        if "ResponseTime" in data:
            serverledge_response_time = data["ResponseTime"]
    except Exception:
        pass

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                time.time(),
                name,
                serverledge_response_time,
                node_arch,
                response.status_code,
                policy,
                response_time,
            ]
        )


# --- CLASSI UTENTE ---
#
# wait_time a zero e peso identico come nell'originale: ogni utente invoca la
# propria funzione in modo continuo, così il carico è determinato dalla durata
# delle funzioni e non da attese artificiali.


class PrimenumUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/primenum", json={"params": {}}, name="primenum"
        )


class ReaddiskUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/readdisk", json={"params": {}}, name="readdisk"
        )


class ReadmemoryUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/readmemory", json={"params": {}}, name="readmemory"
        )


class ThreadUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post("/invoke/thread", json={"params": {}}, name="thread")
