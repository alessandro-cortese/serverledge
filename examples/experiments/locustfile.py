"""Locustfile per le trenta funzioni di benchmark.

Estende quello della tesi precedente, che ne copriva nove. La struttura degli
event listener e il formato del CSV sono invariati, cosi' che gli script di
analisi esistenti restino compatibili.

Ogni funzione ha la propria classe utente con peso identico e attesa nulla: il
carico e' quindi determinato dalla durata delle funzioni, non da attese
artificiali. Con trenta utenti, uno per funzione, ciascuna viene invocata di
continuo per tutta la durata dell'esperimento.

Il timeout e' impostato per funzione anziche' globalmente: le durate osservate
sui nodi sperimentali vanno da 1,4 a 126 secondi, e un valore unico
penalizzerebbe le funzioni lente o renderebbe inutile il controllo su quelle
veloci.
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
            csv.writer(f).writerow(
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
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(
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

    # L'architettura che ha eseguito arriva in questo header, impostato dal
    # nodo e propagato dal load balancer. E' il dato che distingue le due
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
        csv.writer(f).writerow(
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


# --- CLASSI UTENTE ---------------------------------------------------------
#
# Il timeout di ciascuna funzione e' circa il triplo della durata osservata sui
# nodi sperimentali: assorbe il cold start e la contesa sotto carico, senza
# rendere il controllo inefficace.


class Base64streamUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/base64stream",
            json={"params": {}},
            name="base64stream",
            timeout=30,
        )


class CompressionUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/compression",
            json={"params": {}},
            name="compression",
            timeout=30,
        )


class DnaVisualisationUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/dna-visualisation",
            json={"params": {}},
            name="dna-visualisation",
            timeout=30,
        )


class DynamichtmlUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/dynamichtml",
            json={"params": {}},
            name="dynamichtml",
            timeout=30,
        )


class GoroutinesUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/goroutines",
            json={"params": {}},
            name="goroutines",
            timeout=30,
        )


class GraphBfsUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/graph-bfs",
            json={"params": {}},
            name="graph-bfs",
            timeout=30,
        )


class GraphMstUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/graph-mst",
            json={"params": {}},
            name="graph-mst",
            timeout=30,
        )


class GraphPagerankUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/graph-pagerank",
            json={"params": {}},
            name="graph-pagerank",
            timeout=30,
        )


class HashingUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/hashing",
            json={"params": {}},
            name="hashing",
            timeout=30,
        )


class JsonparseUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/jsonparse",
            json={"params": {}},
            name="jsonparse",
            timeout=30,
        )


class MatmulUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/matmul",
            json={"params": {}},
            name="matmul",
            timeout=30,
        )


class MutexcontentionUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/mutexcontention",
            json={"params": {}},
            name="mutexcontention",
            timeout=30,
        )


class PointerchaseUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/pointerchase",
            json={"params": {}},
            name="pointerchase",
            timeout=30,
        )


class RandomaccessUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/randomaccess",
            json={"params": {}},
            name="randomaccess",
            timeout=30,
        )


class SortingUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/sorting",
            json={"params": {}},
            name="sorting",
            timeout=30,
        )


class SyscallstormUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/syscallstorm",
            json={"params": {}},
            name="syscallstorm",
            timeout=30,
        )


class TempfileioUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/tempfileio",
            json={"params": {}},
            name="tempfileio",
            timeout=30,
        )


class ThumbnailerUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/thumbnailer",
            json={"params": {}},
            name="thumbnailer",
            timeout=30,
        )


class TwinChacha20User(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/twin-chacha20",
            json={"params": {}},
            name="twin-chacha20",
            timeout=30,
        )


class TwinPrimenumberUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/twin-primenumber",
            json={"params": {}},
            name="twin-primenumber",
            timeout=30,
        )


class TwinReadmemoryUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/twin-readmemory",
            json={"params": {}},
            name="twin-readmemory",
            timeout=30,
        )


class PrimenumberUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/primenumber",
            json={"params": {}},
            name="primenumber",
            timeout=238,
        )


class Chacha20User(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/chacha20",
            json={"params": {}},
            name="chacha20",
            timeout=30,
        )


class ReaddiskUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/readdisk",
            json={"params": {}},
            name="readdisk",
            timeout=30,
        )


class ReadmemoryUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/readmemory",
            json={"params": {}},
            name="readmemory",
            timeout=76,
        )


class ThreadUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/thread",
            json={"params": {}},
            name="thread",
            timeout=275,
        )


class AmdFasterUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/amd_faster",
            json={"params": {}},
            name="amd_faster",
            timeout=30,
        )


class ArmFasterUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/arm_faster",
            json={"params": {}},
            name="arm_faster",
            timeout=30,
        )


class LinpackUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/linpack",
            json={"params": {}},
            name="linpack",
            timeout=78,
        )


class FilehandleUser(HttpUser):
    wait_time = constant(0.0)
    weight = 1

    @task
    def invoke(self):
        self.client.post(
            "/invoke/filehandle",
            json={"params": {}},
            name="filehandle",
            timeout=379,
        )
