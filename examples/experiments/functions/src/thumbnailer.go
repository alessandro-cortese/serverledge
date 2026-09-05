package main

// thumbnailer — adattamento del benchmark 210.thumbnailer di SeBS.
//
// L'originale scarica un'immagine da storage e ne produce una miniatura. Qui
// l'immagine viene generata in-funzione, cosi' da eliminare la dipendenza
// dalla rete che renderebbe la misura non confrontabile con le altre.
//
// Il ridimensionamento usa l'interpolazione bilineare implementata a mano:
// image/draw della libreria standard non offre un ridimensionamento di
// qualita', e una dipendenza esterna richiederebbe di gestire il vendoring
// nel bundle.
//
// Profilo atteso: CPU utente su interi con accesso bidimensionale alla
// memoria — righe contigue ma salti fra una riga e l'altra. E' un pattern
// intermedio fra la scansione sequenziale e l'accesso casuale.

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"runtime"

	"github.com/serverledge-faas/serverledge/serverledge"
)

// generateImage produce un'immagine sintetica con gradienti e texture, in modo
// che la compressione JPEG abbia un contenuto realistico su cui lavorare.
func generateImage(width int, height int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r := uint8((x * 255) / width)
			g := uint8((y * 255) / height)
			b := uint8(((x ^ y) & 0xFF))

			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	return img
}

// resizeBilinear ridimensiona con interpolazione bilineare: ogni pixel di
// destinazione e' la media pesata dei quattro pixel sorgente circostanti.
func resizeBilinear(src *image.RGBA, width int, height int) *image.RGBA {
	dst := image.NewRGBA(image.Rect(0, 0, width, height))

	srcW := src.Bounds().Dx()
	srcH := src.Bounds().Dy()

	xRatio := float64(srcW-1) / float64(width)
	yRatio := float64(srcH-1) / float64(height)

	for y := 0; y < height; y++ {
		sy := float64(y) * yRatio
		y0 := int(sy)
		dy := sy - float64(y0)

		for x := 0; x < width; x++ {
			sx := float64(x) * xRatio
			x0 := int(sx)
			dx := sx - float64(x0)

			c00 := src.RGBAAt(x0, y0)
			c10 := src.RGBAAt(x0+1, y0)
			c01 := src.RGBAAt(x0, y0+1)
			c11 := src.RGBAAt(x0+1, y0+1)

			blend := func(v00, v10, v01, v11 uint8) uint8 {
				top := float64(v00)*(1-dx) + float64(v10)*dx
				bottom := float64(v01)*(1-dx) + float64(v11)*dx

				return uint8(top*(1-dy) + bottom*dy)
			}

			dst.Set(x, y, color.RGBA{
				R: blend(c00.R, c10.R, c01.R, c11.R),
				G: blend(c00.G, c10.G, c01.G, c11.G),
				B: blend(c00.B, c10.B, c01.B, c11.B),
				A: 255,
			})
		}
	}

	return dst
}

func myHandler(params map[string]interface{}) (interface{}, error) {
	sourceWidth := 2048
	if val, ok := params["width"].(float64); ok {
		sourceWidth = int(val)
	}

	sourceHeight := 1536
	if val, ok := params["height"].(float64); ok {
		sourceHeight = int(val)
	}

	thumbnails := 12
	if val, ok := params["thumbnails"].(float64); ok {
		thumbnails = int(val)
	}

	source := generateImage(sourceWidth, sourceHeight)

	totalBytes := 0

	for i := 0; i < thumbnails; i++ {
		thumb := resizeBilinear(source, 320, 240)

		var buffer bytes.Buffer
		if err := jpeg.Encode(&buffer, thumb, nil); err != nil {
			return nil, err
		}

		totalBytes += buffer.Len()
	}

	return map[string]interface{}{
		"message":      "Thumbnailing completed",
		"source":       []int{sourceWidth, sourceHeight},
		"thumbnails":   thumbnails,
		"output_bytes": totalBytes,
		"arch":         runtime.GOARCH,
	}, nil
}

func main() {
	serverledge.Start(myHandler)
}
