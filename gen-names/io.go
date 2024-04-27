package gen

import (
	"encoding/csv"
	"log"
	"os"
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
