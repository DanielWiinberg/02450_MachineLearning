import toolbox_02450
import numpy as np

version = toolbox_02450.__version__
print(version)

v1 = [0.0247, -0.0388, -0.3288, -0.2131, 0.0477, -0.4584, 0.2683, -0.0838, -0.5020, -0.0200, -0.3091, -0.2588, 0.3714]
m = []

projection_v1 = 0
for v in v1:
  if(v < 0):
    projection_v1 += 1*v
    m.append(1)
  else:
    projection_v1 += 0.1*v
    m.append(0.1)


s1 = np.dot(v1, m)
s2 = np.dot(m, v1)

print('v1', projection_v1)
print(s1, s2)