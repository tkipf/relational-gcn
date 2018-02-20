1. Description: The BGS dataset was created by the British Geological Survey and describes geological measurements in Great Britain (http://data.bgs.ac.uk/). It was first used in~\cite{deVries2013 to predict the lithogenesis property of named rock units. The dataset contains around 150 named rock units with a lithogenesis, from which we used the two largest classes.

2. ML taks: classification

3. Number of instances: 146

4. Original source: BGS

5. Linked to: BGS

6. Target variables:
	-"label_lithogenesis" (classification)
	-label_theme" (classification)


7. Stratified data split (training/test):
	-label_lithogenesis: TrainingSet(lith).tsv (80%) and TestSet(lith).tsv (20%)
	-label_theme: TrainingSet(theme).tsv (80%) and TestSet(theme).tsv (20%)



8. labels distribution
 - label_lithogenesis: label = GLACI =53; label = FLUV =93
 - label_theme: label = BEDRCK =857; label = SPRFCL = 118