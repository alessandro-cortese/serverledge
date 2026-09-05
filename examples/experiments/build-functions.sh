#!/usr/bin/env bash
#
# Compila gli handler Go per entrambe le architetture e produce i bundle .tar
# nel formato che il runtime go125 si aspetta.
#
#     ./build-functions.sh [directory_sorgenti] [directory_output]
#
# Il formato è quello dei bundle della tesi precedente: un archivio TAR che
# contiene uno ZIP con due binari, handler_amd64 e handler_arm64.
# L'entrypoint del runtime scompatta lo zip, legge "uname -m" e seleziona il
# binario corrispondente: è questo meccanismo a rendere la stessa funzione
# eseguibile su x86 e su ARM.
#
# Va eseguito dalla radice del repository Serverledge, perché gli handler
# importano il package serverledge dal modulo locale.

set -euo pipefail

SRC="${1:-functions/src}"
OUT="${2:-functions/bundles}"

command -v go >/dev/null 2>&1 || {
    echo "go non trovato nel PATH"
    exit 1
}

command -v zip >/dev/null 2>&1 || {
    echo "zip non trovato: installalo con 'sudo apt-get install zip'"
    exit 1
}

mkdir -p "$OUT"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

for source in "$SRC"/*.go; do
    [ -e "$source" ] || continue

    name="$(basename "$source" .go)"

    echo "=== $name"

    # CGO disabilitato: il binario deve essere statico. L'immagine go125 è
    # basata su Alpine, che usa musl invece di glibc, e un eseguibile linkato
    # dinamicamente a glibc fallirebbe con un "not found" che sembra
    # inspiegabile — il file c'è, manca il linker.
    for arch in amd64 arm64; do
        echo -n "    $arch ... "
        CGO_ENABLED=0 GOOS=linux GOARCH="$arch" \
            go build -trimpath -ldflags="-s -w" \
            -o "$WORK/handler_${arch}" "$source"
        echo "$(du -h "$WORK/handler_${arch}" | cut -f1)"
    done

    (
        cd "$WORK"
        rm -f "${name}.zip"
        zip -q "${name}.zip" handler_amd64 handler_arm64
        tar cf "${name}.tar" "${name}.zip"
    )

    mv "$WORK/${name}.tar" "$OUT/"
    rm -f "$WORK/handler_amd64" "$WORK/handler_arm64" "$WORK/${name}.zip"

    echo "    -> $OUT/${name}.tar"
done

echo
echo "Bundle prodotti:"
ls -la "$OUT"/*.tar
