def vowel(str):
    vwl=0
    for s in str:
        if s in "aeiouAEIOU":
            vwl +=1
    return vwl


print(vowel("hii"))