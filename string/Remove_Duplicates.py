def Remove_Duplicates(str):
    final_str=[]
    for s in str:
         if s not in final_str:
              final_str.append(s)
    return "".join(final_str).replace(" ", "")   # convert list to string

print(Remove_Duplicates("hi how r u"))
