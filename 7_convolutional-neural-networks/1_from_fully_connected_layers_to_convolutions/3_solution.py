# translation variance means that a part of the image being in different parts of the image
# should have the same meaning. This may not be true in some cases. There are many examples to 
# this. For example in computer vision for detecting vehicles on road. A bigger vehicle far away 
# and a closer, smaller vehicle might look similar on the image, albeit in different locations. 
# But they should have very different interpretations from the model, as the closer vehicle 
# should elicit some response.