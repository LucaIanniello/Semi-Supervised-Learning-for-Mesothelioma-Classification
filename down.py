import gdown
import os

def scarica_cartella_google_drive():
    # URL della cartella Google Drive
    url = "https://drive.google.com/drive/folders/1fm7rYntcwk72huxIKXlJ4qTYkkjcMTl_?usp=share_link"
    
    # ID della cartella estratto dall'URL
    folder_id = "1fm7rYntcwk72huxIKXlJ4qTYkkjcMTl_"
    
    # Crea una cartella locale per i download
    cartella_download = "download_google_drive"
    if not os.path.exists(cartella_download):
        os.makedirs(cartella_download)
    
    # Cambia nella cartella di download
    os.chdir(cartella_download)
    
    try:
        print("Inizio download della cartella...")
        # Scarica l'intera cartella
        gdown.download_folder(url, quiet=False, use_cookies=False)
        print("Download completato!")
        
    except Exception as e:
        print(f"Errore durante il download: {e}")
        # Prova con l'ID diretto se l'URL completo non funziona
        try:
            print("Tentativo con ID cartella...")
            gdown.download_folder(id=folder_id, quiet=False, use_cookies=False)
            print("Download completato con ID!")
        except Exception as e2:
            print(f"Errore anche con ID: {e2}")

if __name__ == "__main__":
    scarica_cartella_google_drive()
