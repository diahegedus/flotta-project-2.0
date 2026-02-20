import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import io
from datetime import datetime

# --- 1. HITELÉSÍTÉSI RENDSZER ---
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
        st.error("Hibás hitelesítési adatok.")
        return False
    else:
        return True

if check_password():
    # --- KONFIGURÁCIÓ ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        st.error("Kritikus hiba: API kulcs nem található.")
        st.stop()

    DB_FILE = "masterdata_forgalmi.csv"

    # --- MASTER DATA KEZELÉS ---
    def load_data():
        if os.path.exists(DB_FILE):
            return pd.read_csv(DB_FILE)
        return pd.DataFrame(columns=[
            "Alvazszam", "Rendszam", "Vevo_Tulajdonos", "Elado", "Brutto_Vetelar", 
            "Teljesitmeny_kW", "Hengerurtartalom_cm3", "Elso_forgalomba_helyezes", 
            "Dokumentum_Tipus", "Feldolgozasi_Statusz", "Utolso_Modositas_Ideje"
        ])

    def save_data(df):
        df.to_csv(DB_FILE, index=False)

    def upsert_record(new_data_dict):
        df = load_data()
        alvaz = new_data_dict.get("Alvazszam")
        
        # Technikai mezők kitöltése
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data_dict["Utolso_Modositas_Ideje"] = now_str
        new_data_dict["Feldolgozasi_Statusz"] = "Kész" # Később ide jöhet a 80%-os AI validációs logika

        if alvaz and str(alvaz).lower() != "null":
            if alvaz in df["Alvazszam"].values:
                idx = df.index[df['Alvazszam'] == alvaz][0]
                for key, value in new_data_dict.items():
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
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = """
            Elemezd a dokumentumot (forgalmi vagy számla) és add vissza az adatokat JSON formátumban:
            - Dokumentum_Tipus: "Forgalmi" vagy "Számla"
            - Alvazszam: 17 karakteres VIN (Kritikus adat)
            - Rendszam: Rendszám (ha szerepel)
            - Vevo_Tulajdonos: Vevő neve vagy C.1 kód alatti név
            - Elado: Eladó neve (csak számla esetén)
            - Brutto_Vetelar: Bruttó végösszeg (csak számla esetén, számértékként)
            - Teljesitmeny_kW (P.2), Hengerurtartalom_cm3 (P.1), Elso_forgalomba_helyezes (B)

            Csak a nyers JSON-t add vissza, egyéb szöveg nélkül.
            """
            pdf_part = {"mime_type": "application/pdf", "data": uploaded_file.getvalue()}
            response = model.generate_content([prompt, pdf_part])
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except:
            return None

    # --- FELÜLET ---
    st.title("🚗 Flotta Admin: Master Data & Napi Riport")
    
    with st.sidebar:
        st.write(f"Bejelentkezve: **{st.secrets['credentials']['username']}**")
        if st.sidebar.button("Kijelentkezés"):
            if "password_correct" in st.session_state:
                del st.session_state["password_correct"]
            st.rerun()

    # --- 1. SZEKCIÓ: TÖMEGES FELDOLGOZÁS ---
    st.subheader("1. Dokumentumok feldolgozása (Pufferelés)")
    uploaded_files = st.file_uploader("Válassza ki a PDF fájlokat (forgalmi vagy számla)", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f"{len(uploaded_files)} fájl feldolgozásának indítása"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            new_recs, updated_recs, errors = 0, 0, 0

            for i, file in enumerate(uploaded_files):
                status_placeholder.text(f"Fájl feldolgozása: {file.name}")
                data = process_document_with_gemini(file)
                
                if data:
                    status = upsert_record(data)
                    if status == "new": new_recs += 1
                    elif status == "update": updated_recs += 1
                    else: errors += 1
                else:
                    errors += 1
                
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_placeholder.success(f"Feldolgozás befejezve. Új rekord: {new_recs} | Frissített: {updated_recs} | Hiba: {errors}")

    st.divider()

    # --- 2. SZEKCIÓ: NAPI ADATKÖZLŐ (REPORTING FLOW) ---
    st.subheader("📅 2. Napi Zárás és Adatközlő Export")
    st.markdown("A mai napon feldolgozott, biztosító és BBO felé továbbítandó 'Kész' státuszú tételek kigyűjtése.")
    
    df_admin = load_data()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if not df_admin.empty:
        # Szűrés a mai napra módosított adatokra
        # (Biztosítjuk, hogy a NaN értékeket is lekezeljük string műveletnél)
        df_admin['Utolso_Modositas_Ideje'] = df_admin['Utolso_Modositas_Ideje'].fillna('')
        df_daily = df_admin[df_admin["Utolso_Modositas_Ideje"].str.startswith(today_str)]
        
        if not df_daily.empty:
            st.info(f"Ma feldolgozott és rögzített tételek száma: **{len(df_daily)} db**")
            
            # Adatközlő Excel generálása
            output_daily = io.BytesIO()
            with pd.ExcelWriter(output_daily, engine='openpyxl') as writer:
                df_daily.to_excel(writer, index=False, sheet_name='Napi_Betoltes')
            
            # Dátumozott fájlnév (pl. Biztosito_Betoltes_20260220.xlsx)
            file_date = today_str.replace("-", "")
            
            st.download_button(
                label=f"📥 Napi Adatközlő Letöltése (Biztosito_Betoltes_{file_date}.xlsx)",
                data=output_daily.getvalue(),
                file_name=f'Biztosito_Betoltes_{file_date}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                type="primary" # Kiemelt gomb
            )
        else:
            st.info("Ma még nem történt dokumentum-feldolgozás, nincs mit jelenteni.")
    else:
        st.info("A Master Data adatbázis még üres.")

    st.divider()
    
    # --- 3. SZEKCIÓ: MASTER DATA MEGJELENÍTÉSE ---
    with st.expander("🗄️ Teljes Master Data (Központi Járműnyilvántartás) megtekintése"):
        if not df_admin.empty:
            st.dataframe(df_admin, use_container_width=True, hide_index=True)
            
            db_output = io.BytesIO()
            with pd.ExcelWriter(db_output, engine='openpyxl') as writer:
                df_admin.to_excel(writer, index=False, sheet_name='Master_Data')
            
            st.download_button(
                label="Teljes Master Data exportálása",
                data=db_output.getvalue(),
                file_name='master_data_teljes.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
