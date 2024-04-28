package main

import (
	"fmt"

	gen "github.com/eissana/gogen/gen-names"
	nn "github.com/eissana/gograd/neural-network"

	"gonum.org/v1/plot/vg"
)

const (
	epochs = 10
)

func main() {
	layerParams := []nn.LayerParam{
		// Output layer with 28 neuron.
		// Linear output layer. Will apply softmax later on output.
		nn.MakeLayerParam(gen.NumChars, nil),
	}
	// Creates a neural network with input size of 28.
	model := nn.MakeNeuralNetwork(gen.NumChars, layerParams)

	// Reduce batchSize to speed up the process.
	batchSize := 100

	inputs, labels := gen.GetRecords(gen.ReadNames("data/names.txt"), batchSize)
	trainingParam := nn.TrainingParam{
		Epochs:                  epochs,
		Regularization:          0.0, // no regularization
		ClassificationThreshold: 0.5,
		LearningRate:            5,
	}
	// Trains the model and returns losses and scores.
	losses, scores := model.Train(inputs, labels, trainingParam)

	// Computes the accuracy of the model.
	accuracy := nn.Accuracy(scores, labels, trainingParam)
	fmt.Printf("Loss: %3.4f, Accuracy: %3.0f%%\n", losses[len(losses)-1], 100*accuracy)

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
