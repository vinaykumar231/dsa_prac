def non_repeating_character(str):
    seen={}
    for s in str:
        if s  in seen:
            seen[s] +=1
        else:
            seen[s] =1
    for s in seen:
        if seen[s]==1:
            return s 
    return None
print(non_repeating_character("fwfew"))