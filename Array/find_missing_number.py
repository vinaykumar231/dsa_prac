def find_missing_number(arr):
    n = len(arr) + 1  # Since one number is missing
    total_sum = n * (n + 1) // 2  # Sum of first n natural numbers
    arr_sum = sum(arr)  # Sum of elements in the given array
    return total_sum - arr_sum  # The missing number

# Example Usage:
arr = [1, 2, 4, 5, 6]
print("Missing number:", find_missing_number(arr))  # Output: 3
