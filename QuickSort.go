package main

import (
	"fmt"
)

func main() {
	a := []int{7, 6, 5, 4, 3, 2, 1, 9, 8}
	quickSort(a)
	fmt.Println(a)
}

func swap(slice []int, i, j int) {
	temp := slice[i]
	slice[i] = slice[j]
	slice[j] = temp
}

func partition(slice []int) int {
	part := (len(slice)) / 2
	end := len(slice) - 1
	swap(slice, end, part)
	i := 0
	for j := 0; j < end; j++ {
		if slice[end] > slice[j] {
			swap(slice, i, j)
			i++
		}
	}
	swap(slice, i, end)
	return part
}

func quickSort(slice []int) {
	part := partition(slice)
	upper := slice[:part]
	lower := slice[part:]
	if len(upper) > 1 {
		quickSort(upper)
	}
	if len(lower) > 1 {
		quickSort(lower)
	}
}
