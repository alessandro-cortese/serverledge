package main

// syscallstorm — SINTETICA, gruppo "riempimento".
//
// Scopo: occupare la regione ad alto cpu_kernel_delta_ms. Esegue un numero
// molto elevato di chiamate di sistema leggere, senza calcolo in spazio utente
// e senza scrittura su disco.
//
// Profilo atteso: CPU kernel dominante, CPU utente bassa, page fault bassi.
// Nessuna delle funzioni esistenti occupa questa regione: filehandle apre e
// chiude file, ma paga anche il costo del filesystem.

import (
	"os"
	"runtime"
	"syscall"

	"github.com/serverledge-faas/serverledge/serverledge"
)

func myHandler(params map[string]interface{}) (interface{}, error) {
	iterations := 2 * 1000 * 1000
	if val, ok := params["iterations"].(float64); ok {
		iterations = int(val)
	}

	pid := 0
	var usage syscall.Rusage
	var stat syscall.Stat_t

	// Tre chiamate diverse per non far collassare la misura su un unico
	// percorso del kernel: una banale, una che legge contatori interni, una
	// che attraversa il filesystem virtuale.
	for i := 0; i < iterations; i++ {
		pid = os.Getpid()

		if i%3 == 0 {
			_ = syscall.Getrusage(syscall.RUSAGE_SELF, &usage)
		}

		if i%7 == 0 {
			_ = syscall.Stat("/proc/self/stat", &stat)
		}
	}

	return map[string]interface{}{
		"message":     "Syscall storm completed",
		"iterations":  iterations,
		"pid":         pid,
		"voluntary":   usage.Nvcsw,
		"involuntary": usage.Nivcsw,
		"arch":        runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
