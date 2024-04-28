package gen

import (
	nn "github.com/eissana/gograd/neural-network"
)

// NeuralNetwork embeds nn.NeuralNetwork to extend it and override Loss function.
type NeuralNetwork struct {
	*nn.NeuralNetwork
}

func MakeNeuralNetwork(layerParams []nn.LayerParam) NeuralNetwork {
	out := NeuralNetwork{}
	out.NeuralNetwork = nn.MakeNeuralNetwork( /*inputSize=*/ NumChars, layerParams)
	return out
}

func (n *NeuralNetwork) Loss(labels, scores [][]*nn.Value, trainingParam nn.TrainingParam) *nn.Value {
	loss := nn.MakeValue(0.0)
	for i, score := range scores {
		label := labels[i]
		loss = loss.Add(n.loss(label, score))
	}
	loss = loss.Div(nn.MakeValue(float64(len(labels))))
	return loss
}

func atoi(c byte) int {
	switch {
	case c == begin:
		return 0
	case c == end:
		return NumChars - 1
	case c >= 'a' && c <= 'z':
		return int(c - 'a')
	default:
		break
	}
	return -1
}

// Implements softmax cross-entropy loss function for a single label and score.
// Note that label and score are embeddings are single chars (a bigram).
func (n *NeuralNetwork) loss(label, score []*nn.Value) *nn.Value {
	// Normalize scores in place.
	normalize(score)
	out := nn.MakeValue(0.0)
	for i, s := range score {
		pos := label[i].Mul(s.Log())
		neg := nn.MakeValue(1.0).Sub(label[i]).Mul(nn.MakeValue(1.0).Sub(s).Log())
		// -y * log(p) - (1-y)*log(1-p)
		out = out.Sub(pos.Add(neg))
	}
	return out
}

func normalize(score []*nn.Value) {
	sum := nn.MakeValue(0.0)
	for _, s := range score {
		sum = sum.Add(s)
	}
	for k := range score {
		score[k] = score[k].Div(sum)
	}
}
