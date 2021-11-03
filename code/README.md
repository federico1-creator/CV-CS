## Python code of the project.
### README with the files in the directory, with a brief description of the different files for the project.

<p>
Directory: ImageProcessing/

Image_processing e utils_segm.py: sono due script python che permettono di visualizzare i vari filtri applicati sulle immagini. Usati nella prima parte per valutare visivamente i risultati della segmentazione dei diversi vestiti indossati.
________________________________________

Directory: Segmentation/

Semantic models DEF.ipynb: sistema di allenamento e realizzazione delle maschere dei vari vestiti in bianco e nero e valutazione delle metriche. I modelli calcolati durante l’allenamento sono salvati dentro alla directory saved_models.

Deeplab ‘people’.ipynb: utilizzo della rete pre allenata deeplab per estrarre e localizzare la persona da un’immagine esterna inoltre si effettua il posizionamento della persona al centro e su sfondo bianco.
________________________________________

Directory: geom_transf/

affineTransformation.py: contiene il codice che effettua la trasformazione affine.

…

________________________________________

Directory: Retrieval/
Notebook, tabelle .csv, pesi delle reti allenati che sono serviti per il sistema di retrieval.

ass.json: file json che contiene un dizionario con la relazione tra il nome della classe presente nel dataset ed il valore numerico assegnato ad essa. Durante le fasi di retrieval viene utilizzato il valore numerico. Può essere utile nel caso si ri voglia convertire il dato.

dati-triplet.csv: lista del percorso di tutte le immagini del dataset (upper body), con assegnata la classe numerica a cui appartengono.

Dataframe_Triplet_84classi.ipynb: manipolazione dei dati per ottenere dati-triplet.csv.

backup.csv: uguale al file precedenti, ma invece di avere assegnato la classe numerica essa viene indicata con il suo nome completo.

tot_df_TL.csv: tabella contenente per ogni immagine del dataset (indicata con il suo percorso assoluto nel drive) un elemento positivo (stessa classe) ed uno negativo (classe diversa). Per andare a valutare le triplette della triplet loss.

TripletLoss 2_training.ipynb: notebook per la creazione del dataframe contenuto in tot_df_TL.csv.

TripletLoss 2°metodo extraction.ipynb: partendo da tot_df_TL.csv allenare l’autoencoder con la triplet loss.

Resnet18 weights [0:8].ipynb: estraggo e salvo i pesi di una resnet18 per utilizzarla come benchmark. Estraggo solamente i primi 8 layers.

Retrieval 3b_vestiti ricos/Superv DEF: notebook in cui vi è l’implementazione del sistema di retrieval con la parte di allenamento e valutazione dei risultati quantitativi e qualitativi.

</p>