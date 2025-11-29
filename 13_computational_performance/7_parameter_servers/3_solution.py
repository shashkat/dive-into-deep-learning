# A fault tolerance mechanism that pytorch's torchrun recommends using is to save the model 
# parameters and other information like optimizer state, epochs run etc every few epochs. Hence, 
# if a particular process fails due to some reason, we can stop training in all the processes 
# gracefully and start again from the last saved snapshot of the model and other information.
# This is exactly what torchrun does.