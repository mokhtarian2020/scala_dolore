# SCALA DOLORE — Documento di Passaggio di Consegne (Italiano)

**Progetto:** SCALA DOLORE — Sistema di Valutazione del Dolore basato su Intelligenza Artificiale  
**Sistema Padre:** CATO MAIOR  
**Data Passaggio:** Maggio 2026  
**Redatto da:** Sviluppatore Uscente  

---

## 1. Panoramica del Progetto

SCALA DOLORE è un sistema clinico di intelligenza artificiale per la **valutazione non verbale del dolore**, pensato principalmente per pazienti anziani affetti da demenza che non sono in grado di comunicare verbalmente il proprio dolore. Il sistema analizza le espressioni facciali tramite un modello di deep learning e restituisce un punteggio PSPI (Pictorial Scale of Pain Intensity), permettendo ai clinici di misurare e documentare oggettivamente il livello di dolore del paziente.

Il progetto fa parte della piattaforma clinica più ampia **CATO MAIOR**, descritta in dettaglio nel documento `documents/Specifiche App Terapia Dolore.txt`.

---

## 2. Base di Ricerca

Il modello AI si fonda sull'articolo scientifico pubblicato:

> **"Unobtrusive Pain Monitoring in Older Adults with Dementia using Pairwise and Contrastive Training"**  
> IEEE Xplore DOI: [10.1109/...](https://ieeexplore.ieee.org/document/9298886)

Il principio chiave: invece di classificare il dolore da una singola immagine, il modello confronta un **frame target** (espressione attuale) con un **frame di riferimento** (lo stesso paziente con espressione neutra). Questo approccio a coppie rende il sistema più robusto e specifico per ogni paziente.

---

## 3. Struttura del Progetto

```
project/
├── pain_detector.py            # Motore AI principale — inizia da qui
├── clinical_backend.py         # API REST FastAPI (uso clinico)
├── simple_backend.py           # API FastAPI semplificata (sviluppo/test)
├── frontend_server.py          # Server HTTP leggero per l'interfaccia web
├── test.py                     # Script di test base con frame di esempio
├── compare_models.py           # Confronta i due checkpoint preaddestrati
├── detailed_analysis.py        # Strumento di analisi dettagliata immagini
├── test_example_images.py      # Test con immagini in example_frames/
├── start_backend.sh            # Script shell per avviare il backend (ROTTO — vedi §8)
├── standard_face_68.npy        # CRITICO: posizioni medie 68 landmark facciali
├── requirements.txt            # Dipendenze Python
│
├── models/
│   └── comparative_model.py    # Rete neurale ConvNetOrdinalLateFusion
│
├── face_alignment/             # Copia locale della libreria Face Alignment Network (FAN)
│   ├── api.py                  # API principale FAN
│   ├── models.py               # Definizioni modelli FAN
│   ├── utils.py                # Utilità FAN
│   └── detection/              # Rilevatori facciali (S3FD, dlib, cartella)
│
├── checkpoints/                # Pesi modello preaddestrati
│   ├── 50342566/50343918_3/model_epoch4.pt   # UNBC + UofR (40 output) ← PRIMARIO
│   └── 59448122/59448122_3/model_epoch13.pt  # Solo UNBC (7 output)
│
├── backend/
│   └── image_processor.py      # Utilità di pre-elaborazione immagini
│
├── database/
│   └── welodge_connector.py    # Connettore database SQLite
│
├── reports/
│   └── pdf_generator.py        # Generatore di report PDF (ReportLab)
│
├── frontend/
│   ├── clinical_interface.html # Interfaccia web clinica (flusso multi-step)
│   └── index.html              # Interfaccia demo semplice
│
├── example_frames/             # Immagini di esempio per test
├── pretrained/                 # Pesi preaddestrati FAN (attualmente vuota — vedi §8)
├── documents/                  # Specifiche di progetto in italiano
└── docs/images/                # Immagini per la documentazione
```

---

## 4. Come Funziona l'AI — Passo per Passo

1. **Input**: Due immagini dello stesso paziente — un *riferimento* (viso neutro) e un *target* (viso da valutare).
2. **Rilevamento del viso**: Il modello `FaceAlignment` (FAN) con backend S3FD rileva i volti ed estrae 68 landmark facciali.
3. **Allineamento**: Ogni immagine viene allineata tramite:
   - **Trasformazione di similarità** (rotazione/scala) che allinea gli anchor di occhi e bocca.
   - **Trasformazione affine piecewise** che deforma 31 landmark chiave verso un template standard (`standard_face_68.npy`).
4. **Pre-elaborazione**: Conversione in scala di grigi + normalizzazione istogramma CLAHE → patch 160×160 pixel.
5. **Inferenza del modello** (`ConvNetOrdinalLateFusion`):
   - I patch target e di riferimento passano attraverso lo stesso backbone CNN.
   - Le mappe di feature vengono **sottratte** (target − riferimento), catturando la differenza di espressione.
   - Pooling + strati fully-connected → score PSPI in output.
6. **Punteggio**: Se vengono forniti più frame di riferimento, viene restituita la media di tutte le predizioni.

### Scala del Dolore (PSPI)

| Punteggio | Livello          | Significato Clinico                                  |
|-----------|------------------|------------------------------------------------------|
| 0–1       | Nessun Dolore    | Nessun segno visibile di disagio                     |
| 1–3       | Dolore Minimo    | Leggera tensione facciale                            |
| 3–5       | Dolore Lieve     | Cambiamenti visibili nell'espressione facciale       |
| 5–7       | Dolore Moderato  | Chiari indicatori di dolore, cipiglio, tensione      |
| 7–10      | Dolore Severo    | Significativa distorsione facciale                   |
| 10+       | Dolore Estremo   | Distress massimo, espressione al limite              |

---

## 5. Checkpoint del Modello Preaddestrato

| File | Dati di Addestramento | `num_outputs` | Note |
|------|-----------------------|---------------|------|
| `checkpoints/50342566/50343918_3/model_epoch4.pt` | UNBC-McMaster **+** Università di Regina "Pain in Severe Dementia" | 40 | **Raccomandato per uso clinico** (pazienti con demenza) |
| `checkpoints/59448122/59448122_3/model_epoch13.pt` | Solo UNBC-McMaster | 7 | Adulti sani in generale |

> **Importante**: I soggetti 66, 80, 97, 108, 121 dell'UNBC sono stati esclusi dall'addestramento per evitare data leakage.

---

## 6. Come Avviare il Sistema

### Prerequisiti
- Python 3.6+
- PyTorch 1.6+ (testato con 2.x)
- CUDA 10.2+ (opzionale ma raccomandato; il sistema ricade su CPU se non disponibile)
- Tutti i pacchetti in `requirements.txt`

```bash
pip install -r requirements.txt
```

### Test Rapido
```bash
python test.py                     # Usa il modello UNBC + UofR
python test.py -unbc_only          # Usa il modello solo UNBC
python test.py -test_framerate     # Misura anche i fotogrammi al secondo
```

### Backend API Semplice (sviluppo/test)
```bash
uvicorn simple_backend:app --host 0.0.0.0 --port 8000 --reload
```

### Backend API Clinico Completo
```bash
uvicorn clinical_backend:app --host 0.0.0.0 --port 8001
```

### Interfaccia Web Frontend
```bash
python frontend_server.py          # Serve su http://localhost:3002
```

---

## 7. Endpoint API (Backend Clinico)

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| `POST` | `/api/check_patient_reference` | Verifica se il paziente ha un'immagine di riferimento nel DB |
| `POST` | `/api/save_reference_image` | Carica e salva un'immagine di riferimento per un paziente |
| `POST` | `/api/assess_pain_with_report` | Flusso completo: valuta il dolore + genera report PDF |

Tutti gli endpoint usano `multipart/form-data`. Il campo `patient_id` è obbligatorio in ogni chiamata.

---

## 8. Problemi Noti e Debito Tecnico

### Critici / Bug

1. **`start_backend.sh` fa riferimento a un file inesistente**: Lo script lancia `python main.py` nella cartella `backend/`, ma `backend/main.py` non esiste. Lo script è attualmente rotto. Occorre creare `backend/main.py` oppure modificare lo script per avviare `clinical_backend.py` o `simple_backend.py` dalla root del progetto.

2. **Errore di battitura in `pain_detector.py`**: Il metodo `verify_refenerece_image` (riga ~100) è scritto in modo errato. `clinical_backend.py` lo chiama come `verify_reference_image`, il che causerà un `AttributeError` a runtime. **Soluzione**: rinominare il metodo in `pain_detector.py` in `verify_reference_image`.

3. **La cartella `pretrained/` è vuota**: FAN (Face Alignment Network) tenterà di scaricare i propri pesi da internet al primo avvio. Su macchine senza accesso a internet, questo fallirà silenziosamente. Occorre pre-scaricare i pesi, posizionarli in `pretrained/` e passare il percorso tramite il parametro `fan_checkpoint` di `PainDetector`.

4. **`WelodgeConnector` usa SQLite, non il vero Welodge**: Il connettore in `database/welodge_connector.py` salva i dati in un file SQLite locale (`welodge.db`). **Non è collegato al sistema clinico reale CATO MAIOR / Welodge**. È uno stub per lo sviluppo. L'integrazione reale richiede l'implementazione delle API REST o della connessione al database effettivo di Welodge.

### Sicurezza / Produzione

5. **CORS completamente aperto**: Entrambi i backend usano `allow_origins=["*"]`. Deve essere ristretto alle origini client reali prima di qualsiasi deploy in produzione.

6. **Nessuna autenticazione**: Il file `requirements.txt` include `python-jose[cryptography]` (libreria JWT) e `aiofiles`, indicando che l'autenticazione era pianificata ma **mai implementata**. Tutti gli endpoint API sono completamente non protetti. Qualsiasi richiesta con un `patient_id` valido può accedere ai dati di qualsiasi paziente.

7. **Immagini salvate come BLOB in SQLite**: Salvare dati binari di immagini in SQLite scala male. Per la produzione, le immagini dovrebbero essere salvate su file system o object storage (es. S3/MinIO), con solo i percorsi nel database.

8. **Nessuna validazione del formato `patient_id`**: Il `patient_id` viene passato direttamente nelle query SQL tramite istruzioni parametrizzate (sicuro contro SQL injection), ma non c'è validazione del formato (es. lunghezza, charset, esistenza in CatoMaior).

### Performance

9. **Inferenza single-threaded**: Il modello esegue una predizione alla volta. Per l'uso clinico concorrente, dovrebbe essere introdotta una coda di lavori (es. Celery + Redis).

10. **I frame di riferimento si accumulano**: `PainDetector.add_references()` aggiunge frame a una lista senza mai svuotarla. Se la stessa istanza di `PainDetector` viene riutilizzata su più valutazioni, i frame di riferimento dei pazienti precedenti contamineranno le predizioni. L'istanza deve essere azzerata tra una valutazione e l'altra, oppure deve essere re-istanziata per ogni richiesta.

---

## 9. Cosa È Stato Realizzato (Riepilogo)

- [x] Motore AI principale per la rilevazione del dolore (classe `PainDetector`)
- [x] Pipeline di allineamento facciale e rilevamento landmark (FAN + S3FD)
- [x] Due checkpoint preaddestrati (UNBC+UofR, solo UNBC)
- [x] API REST clinica (FastAPI) con flusso di lavoro per paziente
- [x] Stub database SQLite per riferimenti paziente e valutazioni
- [x] Generazione report PDF (ReportLab) con dati paziente, immagini, scala del dolore
- [x] Interfaccia web clinica (flusso HTML/JS multi-step)
- [x] Utilità di pre-elaborazione immagini (resize, CLAHE, miglioramento qualità)
- [x] Script di utilità per confronto modelli e analisi
- [x] Script di test base con frame di esempio

---

## 10. Cosa Manca Ancora (Lavori Futuri)

### Alta Priorità

- [ ] **Correggere il typo `verify_reference_image`** in `pain_detector.py`
- [ ] **Creare `backend/main.py`** o correggere `start_backend.sh` per puntare all'entry point corretto
- [ ] **Implementare l'autenticazione**: autenticazione JWT con `python-jose` (già nei requirements). Proteggere tutti gli endpoint `/api/*`
- [ ] **Restringere CORS** alle origini client reali
- [ ] **Integrazione reale con Welodge/CatoMaior**: Sostituire lo stub SQLite con chiamate API reali al backend CATO MAIOR. Le specifiche in `documents/Specifiche App Terapia Dolore.txt` descrivono in dettaglio i requisiti di integrazione completi (registrazione paziente, gestione subscription, DMS, notifiche)

### Media Priorità

- [ ] **Pre-includere i pesi FAN preaddestrati** in `pretrained/` per il deploy offline
- [ ] **Azzerare i frame di riferimento tra una valutazione e l'altra** (`pain_detector.ref_frames = []` prima di ogni nuova sessione paziente)
- [ ] **Spostare lo storage delle immagini** fuori da SQLite verso file/object storage
- [ ] **Aggiungere UI per la cronologia delle valutazioni**: Il database registra tutte le valutazioni, ma non esiste UI per visualizzare le valutazioni passate o monitorare l'andamento del dolore nel tempo di un paziente
- [ ] **Supporto video/stream in tempo reale**: Attualmente è supportata solo l'analisi di immagini statiche. Il processamento in tempo reale da webcam o flusso video migliorerebbe significativamente l'usabilità clinica
- [ ] **Logging e monitoraggio degli errori**: Sostituire i `print` e le eccezioni generiche con logging strutturato

### Lungo Termine (secondo i documenti di specifica)

- [ ] **Sviluppo app mobile**: Il file `documents/Specifiche App Terapia Dolore.txt` contiene una specifica completa per un'app mobile companion. Funzionalità chiave: onboarding paziente con validazione Codice Fiscale + numero Tessera Sanitaria (TEAM), notifiche multicanale (push/SMS/email), visualizzazione referti medici, integrazione DMS
- [ ] **Sistema di notifiche**: Gateway notifiche push con FCM/APNs, SMS, email; coda messaggi falliti (Failed Messages Queue)
- [ ] **Integrazione DMS**: Middleware che collega CatoMaior a un Document Management System per l'archiviazione dei report
- [ ] **Docker/containerizzazione**: Pacchettizzare il backend e il modello per un deploy riproducibile
- [ ] **Ottimizzazione performance**: Considerare la sostituzione di FAN/S3FD con un rilevatore facciale più leggero per FPS più elevati su hardware meno potente (attualmente ~9 FPS su RTX 2080 Ti)
- [ ] **Pipeline di ri-addestramento del modello**: Permettere il fine-tuning su nuove popolazioni di pazienti senza riaddestramento completo

---

## 11. Dipendenze Principali e Versioni

| Libreria | Versione | Scopo |
|----------|----------|-------|
| `torch` | 2.10.0 | Inferenza rete neurale |
| `face-alignment` | 1.3.5 | Rilevamento volto + 68 landmark |
| `opencv-python` | 4.13.0.90 | I/O e processing immagini |
| `scikit-image` | 0.26.0 | Trasformazione affine piecewise |
| `fastapi` | (non bloccata) | Framework API REST |
| `reportlab` | ≥3.6.0 | Generazione PDF |
| `python-jose` | (non bloccata) | JWT — pianificato ma non ancora usato |

---

## 12. Riferimenti e Contatti

- **Articolo originale**: https://ieeexplore.ieee.org/document/9298886  
- **Face Alignment Network (FAN)**: https://github.com/1adrianb/face-alignment  
- **Dataset UNBC-McMaster Shoulder Pain**: Contattare i manutentori originali del dataset per l'accesso  
- **Dataset Università di Regina "Pain in Severe Dementia"**: Contattare i manutentori originali del dataset per l'accesso  
- **Specifica CATO MAIOR**: `documents/Specifiche App Terapia Dolore.txt` (in italiano)

---

*In bocca al lupo con il progetto. Il nucleo AI è solido e pronto per l'uso base in produzione; il lavoro principale rimanente riguarda il rafforzamento della sicurezza, l'integrazione con i sistemi clinici reali e lo sviluppo dell'app mobile.*
