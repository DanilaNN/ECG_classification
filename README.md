# ECG_classification

This is simple electrocardiogram classifier written on Python using PyTorch.
Unfortunately there is no Dataset for this classifier in this repo since this Dataset is not public.  
This repo consists of several files:  
	prepare_data_tools.py - tools to prepare Dataset from initial JSON file with ~1000 ECGs. Biosspy library was used for augmentation.  
	main_lib.py - neural networks model description and tests to verify accuracy during training process.  
	run_rnn.py - training recurrent neural networks.  
	run_conv.py - training convolutional neural network.  
	get_metrics.py - calculate precision and recall of trainig models.  
  	
Final training models stores to /Model folder
