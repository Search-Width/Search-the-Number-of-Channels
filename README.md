# Search the Number of Channels for ResNet-18 on CIFAR-10
-------
The code for this part is in the python file search_width.py. The search log is recorded in search_log.txt and the real-time best neural architecture is recorded in function-preserving.png.
# Posttraining
-------
The code for this part is in the python file Posttraining_resnet.py. The best neural architecture of one search is visualized in resnet_channel.png.
# Comparison against the results of original ResNet-18.
-------
The original ResNet-18 can achieve 3.86 test error with 11.54M parameters on CIFAR-10 while the network with the width discovered by our method can achieve 3.40 test error with 9.94M parameters.
