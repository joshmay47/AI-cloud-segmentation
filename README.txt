File descriptions:
	
	ai.py 		= full file used during development for predicting labels (not recommended for use)
	ai_training.py 	= full file used during development for training models (not recommended for use)
	processor 	= full file used during development for preprocessing (not recommended for use)

	ai_lite.py 		= small file with essentials only for users for predicting (recommended)
	processor_lite 	= smaller file with essentials only for users to preprocess (recommended)


Instructions for use:
	Files used will need to be edited for the appropriate locations for other files, such as satellite images path, ibtracs dataset path, labels datapath etc.

	Using the lite versions only results in simpler and faster code than what was used for development, the not recommended code is available as there may be functions that users may find useful.

	The following is example code that results in predicted labels:
	################################################################
		from processor_lite import process
		from ai_lite import predict
		from processor import showPredictions

		halongImages = process("Halong",2002,verbose=True)
		halongPredictions,selections = predict(halongImages,verbose=True)

		for image,prediction,selection in zip(halongImages,halongPredictions,selections):
    			if selection:
        				showPredictions(image,prediction)
	################################################################
	The above code:
		1) imports relevant functions
		2) processes a tc of the name: "Halong" of the year "2002", and prints updates on its current status
		3) predicts labels for the tc, printing updates on its current status and returning the predictions and which images it didn't predict for
		4) shows the processed images overlaid with the predicted label of the corresponding images, skipping if it didn't predict for that timeslot.
