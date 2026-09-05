package main

// dynamichtml — adattamento del benchmark 110.dynamic-html di SeBS.
//
// L'originale genera una pagina HTML da un template Jinja2 con dati casuali.
// Qui si usa text/template della libreria standard, mantenendo la struttura
// del carico: espansione di un template su una collezione di elementi.
//
// Profilo atteso: CPU utente moderata con molte allocazioni di stringhe di
// piccola dimensione. E' un carico da backend web, il piu' leggero
// dell'insieme, e serve a coprire l'estremo inferiore dello spettro.

import (
	"fmt"
	"io"
	"runtime"
	"text/template"

	"github.com/serverledge-faas/serverledge/serverledge"
)

const pageTemplate = `<!DOCTYPE html><html><head><title>{{.Title}}</title></head>
<body><h1>{{.Title}}</h1><ul>
{{range .Rows}}<li id="{{.ID}}"><span class="name">{{.Name}}</span>
<span class="value">{{.Value}}</span></li>
{{end}}</ul></body></html>`

type row struct {
	ID    int
	Name  string
	Value float64
}

type page struct {
	Title string
	Rows  []row
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	rows := 20000
	if val, ok := params["rows"].(float64); ok {
		rows = int(val)
	}

	renders := 40
	if val, ok := params["renders"].(float64); ok {
		renders = int(val)
	}

	tmpl, err := template.New("page").Parse(pageTemplate)
	if err != nil {
		return nil, err
	}

	data := page{Title: "Serverledge benchmark", Rows: make([]row, rows)}

	for i := range data.Rows {
		data.Rows[i] = row{
			ID:    i,
			Name:  fmt.Sprintf("item-%d", i),
			Value: float64(i) * 1.5,
		}
	}

	// L'output viene scartato: interessa il costo dell'espansione del
	// template, non quello della sua memorizzazione.
	for r := 0; r < renders; r++ {
		if err := tmpl.Execute(io.Discard, data); err != nil {
			return nil, err
		}
	}

	return map[string]interface{}{
		"message": "Dynamic HTML completed",
		"rows":    rows,
		"renders": renders,
		"arch":    runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
