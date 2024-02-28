import math

#Problem 9.1
Ir = 1 - (81/306)**2 - (223/306)**2
Iv1 = 1 - (62/170)**2 - (108/170)**2
Iv2 = 1 - (19/136)**2 - (117/136)**2

delta = Ir - (170/306)*Iv1 - (136/306)*Iv2
print(delta)

# Problem 8.2

def logit(w):
  return 1 / (1 + math.exp(-w))

# A
w = -0.51 -0.11 -0.36 -0.28
print(logit(w))

# B
print(logit(-0.51))