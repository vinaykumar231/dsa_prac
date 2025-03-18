def maxprofit(list):
    n=len(list)
    max_profit= 0
    for i in range(1,n):
        if list[i] > list[i-1]:
            max_profit += list[i] - list[i-1]
    return max_profit

print(maxprofit([2,4,5,6,7,89]))