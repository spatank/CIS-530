Run main_generate.py to generate sentences from Jane Austen novels.
The text file corresponding to the Jane Austen corpus must be stored in a folder called data in the same directory as the Python scripts. 
In the event you wish to generate sentences from a different corpus, store the corresponding text file in the data folder, and change the path in line 12 of main_generate.py accordingly.

The following parameters are available for tuning in the text generation neural network: number of epochs, frequency with which training messages are printed, frequency with which training metrics are plotted, size of the hidden layer in the RNN, the number of layers in the neural network, and the learning rate. These hyperparameters may be changed in lines 80-85 in main_generate.py. 