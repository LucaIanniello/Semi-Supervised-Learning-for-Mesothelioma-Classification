### 3.3 Potenziale del Semi-Supervised Learning

Fondamenti teorici dell'apprendimento semi-supervisionato e auto-supervisionato

Prima di analizzare i vantaggi specifici nell'ambito istopatologico, è essenziale chiarire la distinzione concettuale tra apprendimento semi-supervisionato e auto-supervisionato. L'apprendimento semi-supervisionato rappresenta un paradigma di machine learning che combina strategicamente dati etichettati e non etichettati per addestrare modelli di intelligenza artificiale, posizionandosi tra l'apprendimento supervisionato tradizionale e quello completamente non supervisionato. Questo approccio risulta particolarmente vantaggioso quando la raccolta di dati etichettati è proibitivamente costosa o difficile da ottenere, ma sono disponibili grandi quantità di dati non annotati.

L'apprendimento auto-supervisionato (self-supervised), d'altra parte, utilizza i dati stessi per generare segnali supervisori senza fare affidamento su etichette fornite esternamente. In questo paradigma, i modelli apprendono rappresentazioni significative progettando task ausiliari (pretext tasks) che permettono di inferire la "verità di riferimento" direttamente dai dati non etichettati. Questo approccio imita più da vicino il modo in cui gli esseri umani apprendono a classificare gli oggetti, sviluppando rappresentazioni robuste attraverso l'osservazione di strutture e relazioni intrinseche nei dati.

Vantaggi dell'approccio semi-supervisionato

L'apprendimento semi-supervisionato rappresenta una soluzione strategica per affrontare la scarsità di dati annotati in ambito istopatologico, dove il processo di etichettatura richiede competenze altamente specializzate e tempi considerevoli. Voigt et al.  hanno condotto uno studio comparativo approfondito tra metodi di apprendimento semi-supervisionato e auto-supervisionato nel dominio della patologia computazionale, evidenziando come questi approcci possano ridurre significativamente lo sforzo di annotazione mantenendo prestazioni competitive. La ricerca ha confrontato tre metodi all'avanguardia: PAWS come approccio semi-supervisionato, SimCLR come metodo contrastivo auto-supervisionato, e SimSiam come metodo non-contrastivo auto-supervisionato .

I risultati hanno dimostrato che il pretraining con metodi semi- e auto-supervisionati ha generalmente un impatto positivo sulle prestazioni dei classificatori istopatologici, particolarmente evidente in scenari con dati limitati . Tuttavia, contrariamente alle aspettative, PAWS ha mostrato le prestazioni più deboli nonostante utilizzi esplicitamente informazioni di etichetta durante il pretraining, risultando anche il più sensibile alle impostazioni degli iperparametri e all'inizializzazione dei pesi . Al contrario, SimSiam ha dimostrato le migliori prestazioni complessive e la maggiore stabilità quando i pesi dell'encoder vengono aggiornati durante il fine-tuning .

Vantaggi della segmentazione WSI e patch

La segmentazione delle Whole Slide Images (WSI) in patch rappresenta una strategia computazionale fondamentale per l'analisi di immagini istopatologiche ad altissima risoluzione. Voigt et al.  hanno utilizzato patch di dimensioni 96×96 pixel per gestire dataset gigapixel, dimostrando come questa approccio consenta scalabilità computazionale e maggiore rappresentatività dei dati. La suddivisione in patch permette di processare immagini molto grandi con risorse hardware standard, facilitando sia l'addestramento che l'inferenza su larga scala .

Lo studio ha inoltre evidenziato che i modelli addestrati su patch provenienti da diverse WSI tendono ad essere più robusti a variazioni di colorazione e preparazione dei campioni, migliorando la generalizzazione . Tuttavia, è emerso un limite importante: le caratteristiche apprese da un particolare tipo di tessuto sono trasferibili in-dominio solo in misura limitata, suggerendo la necessità di dataset diversificati che includano vari tipi di tessuto per sviluppare encoder applicabili across l'intero dominio istopatologico .

### 4.1 Computer Vision in Histopathology & Multiple Instance Learning in Medical Imaging
L'evoluzione dell'analisi computazionale in istopatologia ha subito una trasformazione radicale negli ultimi anni, passando dall'analisi tradizionale basata su microscopia a sistemi digitali avanzati che integrano artificial intelligence e deep learning. L'integrazione della computer vision in patologia attraverso la digitalizzazione delle slides rappresenta un salto trasformativo nell'evoluzione del campo, offrendo risultati consistenti, riproducibili e obiettivi con velocità e scalabilità sempre crescenti.

Le tecniche di deep learning per WSI analysis affrontano sfide computazionali uniche dovute alle dimensioni gigapixel delle whole slide images, che possono raggiungere da 100 milioni a 10 miliardi di pixels. Gli approcci più efficaci non utilizzano l'intera immagine come input, ma estraggono e utilizzano solo un piccolo numero di patch, tipicamente con dimensioni che variano da 32×32 a 10,000×10,000 pixels, con la maggioranza degli approcci che usa patch di circa 256×256 pixels. Questo approccio di riduzione dell'alta dimensionalità delle WSI può essere visto come una selezione di feature guidata dall'uomo.

I principi del Multiple Instance Learning si basano sul paradigma che ogni WSI viene trattata come una "bag" e le patch estratte da essa come "instances" della bag. In un contesto positivo (bag), esiste almeno un'istanza positiva, mentre in un contesto negativo (bag), tutte le istanze sono negative. Questo framework è particolarmente efficace nel dominio della patologia digitale, poiché le etichette per le whole slide images sono spesso catturate routinariamente, mentre le etichette per patch, regioni o pixel non lo sono. I metodi MIL più recenti tengono conto delle dipendenze globali e locali tra le istanze, utilizzando meccanismi di attenzione e architetture transformer per aggregare le feature delle patch.

Gli approcci semi-supervisionati in medical imaging hanno dimostrato particolare efficacia nell'affrontare la scarsità di dati etichettati nel dominio medico. Il semi-supervised learning può aiutare fornendo una strategia per pre-addestrare una rete neurale con dati non etichettati, seguita da fine-tuning per un task downstream con annotazioni limitate.
Il self-supervised e contrastive learning rappresentano approcci particolarmente promettenti per l'estrazione di rappresentazioni significative da vast archivi istologici. 

I foundation models recenti come UNI e CONCH hanno segnato una svolta significativa nel campo. UNI e CONCH sono notevoli come i primi foundation models addestrati su dataset di patologia interni diversificati attraverso malattie infettive, infiammatorie e neoplastiche, resi apertamente accessibili alla comunità di ricerca. CONCH, un visual-language foundation model sviluppato utilizzando diverse fonti di immagini istopatologiche, testo biomedico e oltre 1.17 milioni di coppie immagine-didascalia, ha dimostrato prestazioni state-of-the-art su 14 benchmark diversi.

Tuttavia, permangono significativi gap nella letteratura attuale. Le sfide principali includono la mancanza di dati etichettati, la variabilità pervasiva tra tessuti e staining, la natura non-booleana dei task diagnostici, e la necessità di approvazione regolatoria. I metodi MIL-based mostrano efficacia per la classificazione e segmentazione istopatologica ma richiedono miglioramenti per la variabilità instance-level e il riconoscimento di piccole regioni, spesso richiedendo vincoli di supervisione aggiuntivi o essendo inclini all'overfitting. La mancanza di benchmark di valutazione standardizzati e protocolli di validazione prospettica rappresenta una limitazione significativa per l'adozione clinica. Inoltre, rimane la sfida dell'interpretabilità e della trasparenza degli algoritmi, essenziali per guadagnare la fiducia dei patologi e facilitare l'integrazione nei workflow clinici

### 4.3 Trident per Segmentazione e Estrazione di Feature

Per l'elaborazione delle immagini istopatologiche e l'estrazione di caratteristiche, è stato utilizzato Trident, un package Python specificamente progettato per il processamento di Whole Slide Images (WSI) tramite foundation models pretrainati. Trident implementa una pipeline robusta di segmentazione tissue-background basata su DeepLabV3 pretrainato sul dataset COCO, che supera i limiti dei tradizionali metodi di thresholding di Otsu o di sogliatura binaria, garantendo una migliore generalizzazione oltre la colorazione H&E e una separazione più efficace del tessuto da rumore e artefatti. 

Il workflow di elaborazione ha previsto inizialmente la segmentazione automatica del tessuto per rimuovere le regioni di background e minimizzare il processamento non necessario, seguita dalla suddivisione delle aree tissutali in patch individuali di dimensioni specificate. Per l'estrazione delle feature, Trident fornisce model factories unificate che consentono di caricare facilmente diversi patch encoders pretrainati. Nel presente studio sono stati utilizzati quattro modelli distinti: ResNet50-ImageNet come baseline tradizionale, UNI e UNIv2 come foundation models generali per la patologia computazionale, e Phikon-v2 come modello specializzato per la predizione di biomarcatori. L'architettura modulare di Trident ha permesso di standardizzare il processo di inferenza attraverso tutti i modelli, facilitando la comparazione diretta delle performance e garantendo riproducibilità nell'estrazione delle rappresentazioni patch-level per le successive analisi downstream.

### 5.2 Feature Extraction

#### Architettura di Estrazione delle Caratteristiche

Per l'estrazione delle caratteristiche dalle immagini istopatologiche, è stata implementata una pipeline di elaborazione basata sui framework **CLAM** e **Trident**. Il processo di feature extraction costituisce un passaggio fondamentale per convertire le patch ad alta risoluzione in rappresentazioni numeriche compatte e informatiche, riducendo significativamente la dimensionalità computazionale necessaria per l'analisi delle Whole Slide Images[1].

La pipeline di estrazione prevede inizialmente la **segmentazione automatica del tessuto** per rimuovere le regioni di background e minimizzare il processamento non necessario, seguita dalla suddivisione delle aree tissutali in patch individuali di dimensioni specificate. CLAM utilizza un approccio di segmentazione basato su thresholding nel spazio colore HSV, mentre Trident implementa una segmentazione più robusta basata su **DeepLabV3 pretrainato sul dataset COCO**, che supera i limiti dei tradizionali metodi di thresholding di Otsu garantendo una migliore generalizzazione oltre la colorazione H&E[1][2].

Per l'**estrazione delle feature**, sono stati utilizzati quattro diversi modelli encoder: **ResNet50-ImageNet** come baseline tradizionale, **UNI** e **UNIv2** come foundation models generali per la patologia computazionale, e **Phikon-v2** come modello specializzato per la predizione di biomarcatori. ResNet50 pretrainato su ImageNet è stato impiegato attraverso il framework CLAM per convertire patch di dimensioni 256×256 pixel in rappresentazioni feature di 1024 dimensioni utilizzando adaptive mean-spatial pooling dopo il terzo blocco residuale della rete[1]. Trident fornisce model factories unificate che consentono di caricare facilmente i diversi patch encoders pretrainati, facilitando la comparazione diretta delle performance e garantendo riproducibilità nell'estrazione delle rappresentazioni patch-level[2].

#### Data Augmentation a Livello Feature

Data la natura gigapixel delle WSI e le conseguenti limitazioni computazionali nell'applicare tecniche di data augmentation tradizionali direttamente sulle immagini complete, è stata adottata una strategia di **augmentation nello spazio delle feature**. Questo approccio consente di evitare la ripetuta estrazione di caratteristiche e permette un'augmentation online durante il training dei modelli MIL, superando le limitazioni dell'augmentation offline a livello di immagine[3][4].

#### Metodi di Estrapolazione

Tra le tecniche di feature space augmentation implementate, particolare attenzione è stata dedicata ai **metodi di estrapolazione** tra campioni nello spazio delle caratteristiche. Come dimostrato da DeVries e Taylor, l'estrapolazione tra campioni in feature space può essere utilizzata per aumentare i dataset e migliorare le performance degli algoritmi di apprendimento supervisionato[3]. Il processo di estrapolazione genera nuovi vettori di feature secondo la formula:

**c' = (cj - ck)λ + cj**

dove c' rappresenta il vettore di contesto sintetico, ci e cj sono vettori di contesto vicini, e λ è una variabile nel range {0, ∞} che controlla il grado di estrapolazione. L'estrapolazione risulta particolarmente vantaggiosa per la generazione di campioni con maggiore variabilità rispetto ai dati già comuni nel dataset, caratteristica essenziale per migliorare la robustezza dei modelli quando i dati di training sono limitati[3].

#### Framework di Augmentation Basato su Diffusion Model

Ispirandosi al framework **AugDiff**, è stata implementata una strategia di augmentation delle feature basata su **Variational Autoencoder (VAE), U-Net e modelli di diffusione** per generare nuovi campioni a livello di feature a partire da quelli estratti[4]. Questo approccio sfrutta la diversità generativa dei diffusion models per migliorare la qualità dell'augmentation delle feature e la proprietà di generazione step-by-step per controllare la conservazione dell'informazione semantica.

Il framework implementato prevede un **processo di training del Denoising AutoEncoder (DAE)** che comprende l'aggiunta di rumore nel processo di diffusione e la predizione del rumore da parte del DAE. Per preservare l'informazione semantica originale, il processo di sampling è suddiviso in due fasi: **K-step Diffusion** e **K-step Denoising**. Nella prima fase viene applicato un processo di diffusione K-step alle feature originali, dove K è inferiore al numero totale di step T. Nella seconda fase, viene impiegato il DAE addestrato per denoising delle feature input per K step, generando versioni aumentate delle feature originali che mantengono le caratteristiche semantiche fondamentali[4].

Questa metodologia di augmentation basata su diffusion models si è dimostrata superiore sulla carta ai metodi tradizionali come Mixup, che spesso producono feature non realistiche, offrendo un framework più efficiente ed efficace per il training MIL con capacità di generazione online e controllo fine della conservazione semantica[4].

[1] Clam
[2] Trident
[3] DATASET-AUGMENTATION-IN-FEATURE-SPACE.pdf
[4] AugDiff
