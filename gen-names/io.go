package gen

import (
	"encoding/csv"
	"log"
	"math/rand"
	"os"
	"time"

	nn "github.com/eissana/gograd/neural-network"
)

const (
	begin       = '^'
	end         = '$'
	avgWordSize = 5  // just a rough estimate.
	NumChars    = 28 // 26 lower case chars plus begin and end chars.
)

// Reads names from a file and return all lines as a slice.
// Each line is a slice of string each representing a word.
func ReadNames(filename string) [][]string {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("failed to open file: %v", err)
	}
	lines, err := csv.NewReader(file).ReadAll()
	if err != nil {
		log.Fatalf("failed to read from file: %v", err)
	}
	return lines
}

// Generates one-hot encoding of all 28 chars.
func getEncodings() [][]*nn.Value {
	out := make([][]*nn.Value, NumChars)
	for i := range out {
		out[i] = make([]*nn.Value, NumChars)
		for j := range out[i] {
			if i == j {
				out[i][j] = nn.MakeValue(1.0)
			} else {
				out[i][j] = nn.MakeValue(0.0)
			}
		}
	}
	return out
}

// Returns the inputs and labels of the lines.
func GetRecords(lines [][]string, batchSize int) ([][]*nn.Value, [][]*nn.Value) {
	numRecords := len(lines)
	batchIndices := getBatchIndices(batchSize, numRecords, time.Now().Unix())

	encodings := getEncodings()
	inputs := make([][]*nn.Value, 0, len(lines))
	labels := make([][]*nn.Value, 0, len(lines))

	for _, i := range batchIndices {
		name := lines[i]
		input := encodings[atoi(begin)]
		for i := range name[0] {
			label := encodings[atoi(name[0][i])]
			inputs = append(inputs, input)
			labels = append(labels, label)
			copy(input, label)
		}
		inputs = append(inputs, input)
		labels = append(labels, encodings[atoi(end)])
	}
	return inputs, labels
}

func getBatchIndices(batchSize, size int, seed int64) []int {
	r := rand.New(rand.NewSource(seed))
	return r.Perm(size)[:batchSize]
}
