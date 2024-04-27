package gen

import (
	"encoding/csv"
	"log"
	"os"
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

// Generates all bigrams from a slice of words.
func GetBigrams(lines [][]string) [][2]byte {
	bigrams := make([][2]byte, 0, len(lines)*avgWordSize)

	for _, name := range lines {
		var ch byte = begin
		for i := range name[0] {
			bigrams = append(bigrams, [2]byte{ch, name[0][i]})
			ch = name[0][i]
		}
		bigrams = append(bigrams, [2]byte{ch, end})
	}
	return bigrams
}
