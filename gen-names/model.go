package gen

import (
	"math/rand"
	"time"

	nn "github.com/eissana/gograd/neural-network"
)

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

func itoa(i int) byte {
	switch {
	case i == 0:
		return begin
	case i == NumChars-1:
		return NumChars - 1
	case i > 0 && i < NumChars-1:
		return byte('a' + i)
	default:
		break
	}
	return 0
}

func GetModelParams() (w [NumChars][NumChars]*nn.Value) {
	for i := range w {
		for j := range w[i] {
			w[i][j] = nn.MakeValue(rand.Float64())
		}
	}
	return w
}

func loss(w [NumChars][NumChars]*nn.Value, input, output byte) *nn.Value {
	i, j := atoi(input), atoi(output)
	wexp := [NumChars]*nn.Value{}
	sum := nn.MakeValue(0.0)
	for k := range w[i] {
		wexp[k] = w[i][k].Exp()
		sum = sum.Add(wexp[k])
	}
	for k := range wexp {
		wexp[k] = wexp[k].Div(sum)
	}
	out := nn.MakeValue(0.0)
	for k := range wexp {
		if k == j {
			out = out.Sub(wexp[k].Log())
		} else {
			out = out.Sub(nn.MakeValue(1.0).Sub(wexp[k]).Log())
		}
	}
	return out
}

func getBatchIndices(batchSize, size int, seed int64) []int {
	r := rand.New(rand.NewSource(seed))
	return r.Perm(size)[:batchSize]
}

func Loss(w [NumChars][NumChars]*nn.Value, bigrams [][2]byte, batchSize int) *nn.Value {
	numBigrams := len(bigrams)
	batchIndices := getBatchIndices(batchSize, numBigrams, time.Now().Unix())

	out := nn.MakeValue(0.0)
	for _, i := range batchIndices {
		bigram := bigrams[i]
		input, output := bigram[0], bigram[1]
		out = out.Add(loss(w, input, output))
	}
	out = out.Div(nn.MakeValue(float64(batchSize)))
	return out
}

func step(w [NumChars][NumChars]*nn.Value, learningRate float64) {
	for i := range w {
		for j := range w[i] {
			data := w[i][j].GetData() - learningRate*w[i][j].GetGrad()
			w[i][j].SetData(data)
		}
	}
}

func resetGrad(w [NumChars][NumChars]*nn.Value) {
	for i := range w {
		for j := range w[i] {
			w[i][j].ResetGrad()
		}
	}
}

func Train(w [NumChars][NumChars]*nn.Value, bigrams [][2]byte, epochs, batchSize int, learningRate float64) []float64 {
	losses := make([]float64, 0, epochs)
	for i := 0; i < epochs; i++ {
		loss := Loss(w, bigrams, batchSize)
		losses = append(losses, loss.GetData())
		resetGrad(w)
		loss.BackPropagate()
		step(w, learningRate)
	}
	return losses
}
