package main

import (
	"fmt"

	gen "github.com/eissana/gogen/gen-names"
	nn "github.com/eissana/gograd/neural-network"

	"gonum.org/v1/plot/vg"
)

func main() {
	bigrams := gen.GetBigrams(gen.ReadNames("data/names.txt"))
	w := gen.GetModelParams()
	epochs, batchSize, learningRate := 80, 100, 50.0
	losses := gen.Train(w, bigrams, epochs, batchSize, learningRate)
	fmt.Printf("final loss: %f\n", losses[len(losses)-1])

	plotter := nn.Plotter{
		Width:  6 * vg.Inch,
		Height: 4 * vg.Inch,
	}
	// Plots the loss function.
	iterations := getX(len(losses))
	plotter.PlotLine(iterations, losses, "results/loss.png")
}

func getX(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
	}
	return x
}
