1. Description: The AM dataset contains information about artifacts in the Amsterdam Museum~\cite{deBoer2012}. Each
artefact in the dataset is linked to other artifacts and details about its production, material, and content. It also
has an artifact category, which serves as a prediction target. We have drawn a stratified random sample of 1,000
instances from the complete dataset. We also removed the \texttt{material} relation, since it highly correlates with the
artifact category.

2. ML task: classificaiton

3. Number of instances: 1000

4. Original source: AM

5. Linked to: AM

6. Target variables
	-"label_cateogory" (classification)


7. Stratified data split (training/test):
	-label_cateogory: TrainingSet.tsv (80%) and TestSet.tsv (20%)

8. labels distribution
    http://purl.org/collections/nl/am/t-14592	55
    http://purl.org/collections/nl/am/t-15459	50
    http://purl.org/collections/nl/am/t-15579	116
    http://purl.org/collections/nl/am/t-15606	81
    http://purl.org/collections/nl/am/t-22503	347
    http://purl.org/collections/nl/am/t-22504	25
    http://purl.org/collections/nl/am/t-22505	86
    http://purl.org/collections/nl/am/t-22506	42
    http://purl.org/collections/nl/am/t-22507	15
    http://purl.org/collections/nl/am/t-22508	56
    http://purl.org/collections/nl/am/t-5504	127
