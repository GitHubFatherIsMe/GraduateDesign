package main

import (
	"fmt"
	"sort"
	"time"
)

func main() {
	start := time.Now()
	nums := []int{-1, 0, 1, 2, -1, -4}
	fmt.Println(threeSum(nums))
	fmt.Println(start.Sub(time.Now()))
}

func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	length := len(nums)
	results := [][]int{}
	for i := 0; i <= length-3; i++ {
		for j := i + 1; j <= length-2; j++ {
			for k := j + 1; k <= length-1; k++ {
				if nums[i]+nums[j]+nums[k] == 0 {
					results = append(results, []int{nums[i], nums[j], nums[k]})
					break
				}
			}
			for {
				if nums[j] == nums[j+1] {
					if j >= length-2 {
						break
					} else {
						j++
					}
				} else {
					break
				}
			}
		}
		for {
			if nums[i] == nums[i+1] {
				if i >= length-2 {
					break
				} else {
					i++
				}
			} else {
				break
			}
		}
	}
	return results
}
