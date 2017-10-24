1. Open Command Prompt
2. Go to Project file path
3. Run pre-processing file "PreProcessing.py" with command
		python PreProcessing.py <Input Dataset Path> <Output filename/path>
	example :
		python PreProcessing.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data PreProcessedfile
4. Use the output file generated in previous step and run the main application BackPropagation.py using  command
		python BackPropagation.py <output filename/path> <training_percent> <max_iteration> <number of hidden layers> <Nuerons in each hidden layers>
	example:
		python BackPropagation.py PreProcessedfile 0.8 100 2 4 2