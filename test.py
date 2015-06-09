import Task
import time

st=time.time()

knnTask=Task.KnnTask("test/result.mat",1)
knnTask.run()

print time.time()-st