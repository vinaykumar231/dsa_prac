def second_max(list):
    first_max=second_max= float("-inf")
    n= len(list)
    for i in range(n):
        if list[i] > first_max:
            second_max=first_max
            first_max=list[i]
            
        elif list[i] > second_max and list[i] != first_max:
            second_max=list[i]
    return second_max

print(second_max([2,4,6,4,7,9,3,33,220]))
