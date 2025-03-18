def is_prime(num):
    if num < 2:
        return False
    for i in range(2, num):
        if num %i ==0:
            return False
        
    return True

all_prime=[]
for i in range(2,20):
    if is_prime(i):
        all_prime.append(i)

print(all_prime)
