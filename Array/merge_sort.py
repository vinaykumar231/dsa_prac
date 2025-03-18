def merge_sort(left, right):
    i =j=0
    sorted_list=[]
    while i < len(left) and j < len(right):

        if left[i] < right[j]:
            sorted_list.append(left[i])
            i +=1
        else:
            sorted_list.append(right[j])
            j+=1
    sorted_list.extend(left[i:])
    sorted_list.extend(right[j:])
    
    return sorted_list


def recursive_find_left_right(list):
    n = len(list)
    
    if n <= 1:
        return list
    
    mid = n // 2
    left = list[:mid]
    right = list[mid:]

    left = recursive_find_left_right(left)
    right = recursive_find_left_right(right)
    
    return merge_sort(left, right)
    
print(recursive_find_left_right([2, 4, 5, 6, 70, 7, 3]))

