def  Count_Vowels_Consonants(str):
    vwl=0
    const=0
    n=len(str)
    for s in str:
        if s in "aeiouAEIOU":
            vwl +=1
        else:
            const +=1
    return {"vowel":vwl , "Consonants": const, "total+length":n}
print(Count_Vowels_Consonants("My Name is lakhan"))