package main

import (
	"fmt"
)

func main() {
	grid := [][]int{{3, 0, 8, 4}, {2, 4, 5, 7}, {9, 2, 6, 3}, {0, 3, 1, 0}}
	sum := maxIncreaseKeepingSkyline(grid)
	fmt.Println(sum)
}

func maxIncreaseKeepingSkyline(grid [][]int) int {
	length := len(grid)
	sum := 0
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			sum += grid[i][j]
		}
	}

	//The skyline viewed from left or right
	lrSkyline := make([]int, length)
	for i, v := range grid {
		lrSkyline[i] = v[i]
		for j, val := range v {
			if v[j] > lrSkyline[i] {
				lrSkyline[i] = val
			}
		}
	}
	_, lrMaxLocation := max(lrSkyline, length)

	//The skyline viewed from top or bottom
	tbSkyline := make([]int, length)
	for i := 0; i < length; i++ {
		tbSkyline[i] = grid[0][i]
		for j := 0; j < length; j++ {
			if tbSkyline[i] < grid[j][i] {
				tbSkyline[i] = grid[j][i]
			}
		}
	}
	_, tbMaxLocation := max(tbSkyline, length)

	gridNew := make([][]int, length)
	for i, _ := range gridNew {
		gridNew[i] = make([]int, length)
	}

	for i, _ := range gridNew[lrMaxLocation] {
		gridNew[lrMaxLocation][i] = tbSkyline[i]
	}
	for i := 0; i < length; i++ {
		gridNew[i][tbMaxLocation] = lrSkyline[i]
	}

	sum1 := fillAndSum(gridNew, tbSkyline, lrSkyline, length)
	return sum1 - sum
}

func max(skyLine []int, length int) (int, int) {
	max := 0
	maxLocation := 0
	for i := 0; i < length; i++ {
		if max < skyLine[i] {
			max = skyLine[i]
			maxLocation = i
		}
	}
	return max, maxLocation
}

func min(gridNew [][]int, i, j int, tbSkyline, lrSkyline []int) int {
	min := 0
	if tbSkyline[j] > lrSkyline[i] {
		min = lrSkyline[i]
	} else {
		min = tbSkyline[j]
	}
	return min
}

func fillAndSum(gridNew [][]int, tbSkyline, lrSkyline []int, length int) int {
	sum := 0
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			if gridNew[i][j] == 0 {
				gridNew[i][j] = min(gridNew, i, j, tbSkyline, lrSkyline)
			}
			sum += gridNew[i][j]
		}
	}
	return sum
}
