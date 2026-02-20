import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import base64
import io

# --- 1. JELSZÓ ELLENŐRZŐ RENDSZER ---
def check_password():
    def password_entered():
        if (
            st.session_state["username"] == st.secrets["credentials"]["username"]
            and st.session_state["password"] == st.secrets["credentials"]["password"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Bejelentkezés")
        st.text_input("Felhasználónév", on_change=password_entered, key="username")
        st.text_input("Jelszó", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Bejelentkezés")
        st.text_input("Felhasználónév", on_change=password_entered, key="username")
        st.text_input("Jelszó", type="password", on_change=password_entered, key="password")
        st.error("😕 Hibás felhasználónév vagy jelszó")
        return False
    else:
        return True

if check_password():
    # --- BEÁLLÍTÁSOK ÉS AI KONFIGURÁCIÓ ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        st.error("❌ API kulcs nem található a Secrets-ben!")
        st.stop()

    DB_FILE = "forgalmi_adatbazis.csv"

    # --- SEGÉDFÜGGVÉNYEK ---
    def load_data():
        if os.path.exists(DB_FILE):
            return pd.read_csv(DB_FILE)
        return pd.DataFrame(columns=[
            "Alvazszam", "Rendszam", "Vevo_Tulajdonos", "Elado", "Brutto_Vetelar", 
            "Teljesitmeny_kW", "Hengerurtartalom_cm3", "Elso_forgalomba_helyezes", "Dokumentum_Tipus"
        ])

    def save_data(df):
        df.to_csv(DB_FILE, index=False)

    def upsert_record(new_data_dict):
        df = load_data()
        alvaz = new_data_dict.get("Alvazszam")
        if alvaz and str(alvaz).lower() != "null":
            if alvaz in df["Alvazszam"].values:
                idx = df.index[df['Alvazszam'] == alvaz][0]
                for key, value in new_data_dict.items():
                    # Csak akkor írjuk felül, ha az új adat nem üres
                    if value and str(value).lower() != "null": 
                        df.at[idx, key] = value
                save_data(df)
                return "update"
            else:
                new_row = pd.DataFrame([new_data_dict])
                df = pd.concat([df, new_row], ignore_index=True)
                save_data(df)
                return "new"
        return "error"

    def process_document_with_gemini(uploaded_file):
        # Automatikus modellválasztás a Google válasza alapján
        try:
            available_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            available_models = ['gemini-1.5-flash', 'gemini-1.5-pro']
            
        preferred_order = ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.5-pro-latest']
        models_to_try = [m for m in preferred_order if m in available_models] or [available_models[0]]

        prompt = """
        Te egy profi flotta adminisztrációs rendszer vagy. Elemezd a csatolt PDF-et (amely forgalmi engedély vagy adásvételi számla).
        Keresd meg és add vissza szigorúan csak JSON formátumban:
        
        - Dokumentum_Tipus: "Forgalmi" vagy "Számla"
        - Alvazszam: 17 karakteres alvázszám (VIN)
        - Rendszam: Forgalmi rendszám (ha van)
        - Vevo_Tulajdonos: Számla esetén a Vevő, forgalmi esetén a Tulajdonos (C.1)
        - Elado: Számla esetén az Eladó neve (egyébként null)
        - Brutto_Vetelar: Számla esetén a bruttó végösszeg (csak a számérték, egyébként null)
        - Teljesitmeny_kW: Forgalmi P.2 kód
        - Hengerurtartalom_cm3: Forgalmi P.1 kód
        - Elso_forgalomba_helyezes: Forgalmi B kód (YYYY.MM.DD)

        Csak a nyers JSON-t írd le, minden más szöveg nélkül!
        """
        
        pdf_part = {"mime_type": "application/pdf", "data": uploaded_file.getvalue()}
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, pdf_part])
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_text)
            except Exception as e:
                continue # Ha hiba van, megy a következő modellre
        return None

    # --- FELÜLET ---
    st.title("📄 Flotta Admin: Tömeges Adatkinyerő Pilot")
    st.markdown("Dokumentumok (forgalmi engedélyek és számlák) automatikus feldolgozása és összefűzése alvázszám alapján.")
    
    # Oldalsáv kijelentkezéssel
    with st.sidebar:
        st.write(f"Bejelentkezve: **{st.secrets['credentials']['username']}**")
        if st.button("Kijelentkezés"):
            if "password_correct" in st.session_state:
                del st.session_state["password_correct"]
            st.rerun()

    # TÖMEGES FELTÖLTÉS
    uploaded_files = st.file_uploader("PDF dokumentumok feltöltése", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f"{len(uploaded_files)} dokumentum feldolgozásának indítása", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            new_count, update_count, error_count = 0, 0, 0

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Feldolgozás alatt ({i+1}/{len(uploaded_files)}): {file.name}")
                
                extracted_data = process_document_with_gemini(file)
                
                if extracted_data:
                    res = upsert_record(extracted_data)
                    if res == "new": new_count += 1
                    elif res == "update": update_count += 1
                    else: error_count += 1
                else:
                    error_count += 1
                
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.success(f"Feldolgozás befejezve! Eredmény: {new_count} új rögzítve | {update_count} frissítve | {error_count} hiba")

    st.divider()
    
    # --- ADATBÁZIS NÉZET ---
    st.subheader("📊 Központi Járműnyilvántartás")
    df_admin = load_data()
    
    if not df_admin.empty:
        st.dataframe(df_admin, use_container_width=True, hide_index=True)
        
        db_output = io.BytesIO()
        with pd.ExcelWriter(db_output, engine='openpyxl') as writer:
            df_admin.to_excel(writer, index=False, sheet_name='Flotta_Lista')
        
        st.download_button(
            label="📥 Teljes adatbázis letöltése (Excel)",
            data=db_output.getvalue(),
            file_name='flotta_nyilvantartas.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info("Az adatbázis jelenleg üres. Tölts fel dokumentumokat a kezdéshez.")
