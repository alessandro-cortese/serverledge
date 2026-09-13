#!/usr/bin/env bash
#
# Catalogo condiviso dei benchmark usati per la raccolta dei profili.
#
# Formato:
#   nome|runtime|sorgente|memoria_MB|handler
#
# Il corpus e' CONGELATO: non modificare i benchmark senza rifare l'intera
# campagna su entrambe le architetture. page_faults_delta, cpu_user_delta_ms e
# cpu_kernel_delta_ms sono quantita' assolute, quindi cambiare la durata di una
# funzione la sposta nello spazio delle feature del clustering.
#
# Questo file e' sorgente di verita' per gcp_collect_profiles_fast.sh e viene
# confrontato con il catalogo inline di gcp_collect_profiles.sh a ogni avvio.

FUNCTIONS=(
    "base64stream|go125|functions/bundles/base64stream.tar|1024|"
    "compression|go125|functions/bundles/compression.tar|1024|"
    "dna-visualisation|go125|functions/bundles/dna-visualisation.tar|1024|"
    "dynamichtml|go125|functions/bundles/dynamichtml.tar|1024|"
    "goroutines|go125|functions/bundles/goroutines.tar|1024|"
    "graph-bfs|go125|functions/bundles/graph-bfs.tar|1024|"
    "graph-mst|go125|functions/bundles/graph-mst.tar|1024|"
    "graph-pagerank|go125|functions/bundles/graph-pagerank.tar|1024|"
    "hashing|go125|functions/bundles/hashing.tar|1024|"
    "jsonparse|go125|functions/bundles/jsonparse.tar|1024|"
    "matmul|go125|functions/bundles/matmul.tar|1024|"
    "mutexcontention|go125|functions/bundles/mutexcontention.tar|1024|"
    "pointerchase|go125|functions/bundles/pointerchase.tar|1024|"
    "randomaccess|go125|functions/bundles/randomaccess.tar|1024|"
    "sorting|go125|functions/bundles/sorting.tar|1024|"
    "syscallstorm|go125|functions/bundles/syscallstorm.tar|1024|"
    "tempfileio|go125|functions/bundles/tempfileio.tar|1024|"
    "thumbnailer|go125|functions/bundles/thumbnailer.tar|1024|"
    "twin-chacha20|go125|functions/bundles/twin-chacha20.tar|1024|"
    "twin-primenumber|go125|functions/bundles/twin-primenumber.tar|1024|"
    "twin-readmemory|go125|functions/bundles/twin-readmemory.tar|1024|"
    "primenumber|go125|primenum.tar|1024|"
    "chacha20|go125|chacha20.tar|1024|"
    "readdisk|go125|readdisk.tar|1024|"
    "readmemory|go125|readmemory.tar|1024|"
    "thread|go125|thread.tar|1024|"
    "amd_faster|go125|amd_fasterV2.tar|1024|"
    "arm_faster|go125|arm_fasterV2.tar|1024|"
    "linpack|python312ml|linpack.py|2048|linpack.handler"
    "filehandle|python314|filehandle.py|1024|filehandle.handler"

    "base64stream-py|python314|multilang/python/base64stream.py|1024|base64stream.handler"
    "compression-py|python314|multilang/python/compression_bench.py|1024|compression_bench.handler"
    "dna-visualisation-py|python314|multilang/python/dna_visualisation.py|1024|dna_visualisation.handler"
    "dynamic-html-py|python314|multilang/python/dynamic_html.py|1024|dynamic_html.handler"
    "float-ops-py|python314|multilang/python/float_ops.py|1024|float_ops.handler"
    "graph-bfs-py|python314|multilang/python/graph_bfs.py|1024|graph_bfs.handler"
    "graph-mst-py|python314|multilang/python/graph_mst.py|1024|graph_mst.handler"
    "graph-pagerank-py|python314|multilang/python/graph_pagerank.py|1024|graph_pagerank.handler"
    "hashing-py|python314|multilang/python/hashing.py|1024|hashing.handler"
    "json-dumps-py|python314|multilang/python/json_dumps.py|1024|json_dumps.handler"
    "jsonparse-py|python314|multilang/python/jsonparse.py|1024|jsonparse.handler"
    "memory-rw-py|python314|multilang/python/memory_rw.py|1024|memory_rw.handler"
    "pointerchase-py|python314|multilang/python/pointerchase.py|1024|pointerchase.handler"
    "randomaccess-py|python314|multilang/python/randomaccess.py|1024|randomaccess.handler"
    "thumbnailer-py|python314|multilang/python/thumbnailer.py|1024|thumbnailer.handler"

    "compression-node|nodejs17ng|multilang/nodejs/compression.js|1024|compression"
    "dynamic-html-node|nodejs17ng|multilang/nodejs/dynamic_html.js|1024|dynamic_html"
    "float-ops-node|nodejs17ng|multilang/nodejs/float_ops.js|1024|float_ops"
    "graph-bfs-node|nodejs17ng|multilang/nodejs/graph_bfs.js|1024|graph_bfs"
    "graph-pagerank-node|nodejs17ng|multilang/nodejs/graph_pagerank.js|1024|graph_pagerank"
    "json-dumps-node|nodejs17ng|multilang/nodejs/json_dumps.js|1024|json_dumps"
    "memory-rw-node|nodejs17ng|multilang/nodejs/memory_rw.js|1024|memory_rw"
    "thumbnailer-node|nodejs17ng|multilang/nodejs/thumbnailer.js|1024|thumbnailer"
)
