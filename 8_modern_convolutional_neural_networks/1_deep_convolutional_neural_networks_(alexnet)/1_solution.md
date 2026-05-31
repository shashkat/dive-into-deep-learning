# part 1
The memory footprint of a certain layer is basically the size of parameters that are associated with that layer. Here, the memory footprint of the convolutions is:
$$11\cdot11\cdot1\cdot96+5\cdot5\cdot96\cdot256+3\cdot3\cdot256\cdot384+3\cdot3\cdot384\cdot384+3\cdot3\cdot384\cdot256$$, which is approximately 3.7 million.
And the memory footprint of the fully connected layers is:
$$6400\cdot4096+4096\cdot4096+4096\cdot10$$, which is approximately 43 million.

Hence, the memory footprint of the fully connected layers is higher by one order of magnitude.

# part 2
Looking at an approximate number of operations performed in the set of convolutional layers and the set of fully connected layers, it seems that the complexity of convolutional layers is higher than that of fully connected layers:

Complexity of convolutional layers: $$11\cdot11\cdot54\cdot54\cdot1\cdot96+5\cdot5\cdot26\cdot26\cdot96\cdot256+3\cdot3\cdot12\cdot12\cdot256\cdot384+3\cdot3\cdot12\cdot12\cdot384\cdot384+3\cdot3\cdot12\cdot12\cdot384\cdot256$$, which is approximately 1 billion.

Complexity of fully connected layers: $$6400\cdot4096+4096\cdot4095+4096\cdot10$$, which is approximately 43 million.

# part 3

During training, the computations are quite heavy for each given piece of data from memory. During inference, the computations are relatively light. Hence, the bottleneck is generally the computation speed of processor in training, whereas during inference, it is the speed at while data can be fetched from memory.

Hence, during training, the memory bandwidth needs to just be sufficient enough to not let the processor starve, which is relatively easy as the computations are heavy in training. The latency doesn't have to be super low, as again, the bottleneck is computation speed and not data transfer. There is also not much benefit of increasing memory size beyong a limit.

During inference, the memory bandwidth needs to be much larger, as each computation is a lot more light, and hence we would want to keep the processor busy and infer as many datapoints as possible. The latency has to be low because else the processor will be starving for some duration for each datapoint. Since the pressure is at memory, if it is bigger, then we benefit more too.




