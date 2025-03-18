def binary_search(lst, value):
    new_list = sorted(lst) 
    print("Sorted list:", new_list)
    left = 0
    right = len(new_list) - 1

    for _ in range(len(new_list)):  
        mid = (left + right) // 2  
        print("Searching for:", value)
        print("Current mid index:", mid, "Value at mid:", new_list[mid])

        if new_list[mid] == value:
            return mid  
        elif new_list[mid] < value:
            left = mid + 1  
        else:
            right = mid - 1  

    return -1  

print(binary_search([2, 32, 3, 311, 31, 45, 66], 31))
