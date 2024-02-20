from toolbox_02450.similarity import similarity

# 4.1
o1 = [[0, 4, 7, 9, 5, 5, 5, 6]]
o3 = [[7, 7, 0, 10, 6, 6, 4, 9]]

print("Cosine: %.4f " % (similarity(o1,o3,'cos')))
print("Jaccard: %.4f " % (similarity(o1,o3,'jac')))
print("SMC: %.4f " % (similarity(o1,o3,'smc')))