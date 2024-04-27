package main

import (
	"fmt"

	gen "github.com/eissana/gograd"
)

func main() {
	words := gen.ReadNames("data/names.txt")
	fmt.Println(words[:3])

}
