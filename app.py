import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import io
import time
from datetime import datetime

# --- 1. HITELÉSÍTÉSI RENDSZER (SZEREPKÖRÖKKEL) ---
def check_password():
    def password_entered():
        user = st.session_state["username"]
        pwd = st.session_state["password"]
        
        # Ellenőrizzük, hogy létezik-e a felhasználó a secrets-ben
        if "users" in st.secrets and user in st.secrets["users"]:
            if st.secrets["users"][user]["password"] == pwd:
                st.session_state["password_correct"] = True
                st.session_state["logged_in_user"] = user
                st.session_state["role"] = st.secrets["users"][user]["role"]
                del st.session_state["password"]
                del st.session_state["username"]
                return
        
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
        st.error("Hibás felhasználónév vagy jelszó.")
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
            df = pd.read_csv(DB_FILE)
            # Biztosítjuk, hogy az új oszlop létezzen a régi adatbázisoknál is
            if "Feltolto_User" not in df.columns:
                df["Feltolto_User"] = "ismeretlen"
            return df
        return pd.DataFrame(columns=[
            "Alvazszam", "Rendszam", "Vevo_Tulajdonos", "Elado", "Brutto_Vetelar", 
            "Teljesitmeny_kW", "Hengerurtartalom_cm3", "Elso_forgalomba_helyezes", 
            "Dokumentum_Tipus", "Feldolgozasi_Statusz", "Utolso_Modositas_Ideje", "Feltolto_User"
        ])

    def save_data(df):
        df.to_csv(DB_FILE, index=False)

    def upsert_record(new_data_dict):
        df = load_data()
        alvaz = new_data_dict.get("Alvazszam")
        
        # Technikai mezők
        new_data_dict["Utolso_Modositas_Ideje"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data_dict["Feldolgozasi_Statusz"] = "Kész"
        new_data_dict["Feltolto_User"] = st.session_state["logged_in_user"] # Eltároljuk, ki töltötte fel!

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
            available_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            available_models = []
            
        preferred_order = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.5-flash']
        models_to_try = [m for m in preferred_order if m in available_models] or (available_models[:1] if available_models else [])

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
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, pdf_part])
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_text)
            except Exception as e:
                continue
        return None

    # --- OLDALSÁV (MINDENKINEK KÖZÖS) ---
    with st.sidebar:
        current_user = st.session_state['logged_in_user']
        current_role = st.session_state['role']
        st.write(f"👤 Felhasználó: **{current_user}**")
        st.write(f"🛡️ Szerepkör: *{current_role.capitalize()}*")
        
        if st.sidebar.button("Kijelentkezés"):
            for key in ["password_correct", "logged_in_user", "role"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # =========================================================
    # 1. ÜGYFÉL NÉZET (UX)
    # =========================================================
    if st.session_state["role"] == "ugyfel":
        st.title("📁 Dokumentum Feltöltő Központ")
        st.markdown("Kérjük, töltse fel ide a gépjárművekhez tartozó forgalmi engedélyeket és számlákat.")
        
        uploaded_files = st.file_uploader("PDF fájlok kiválasztása", type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            if st.button(f"{len(uploaded_files)} fájl beküldése feldolgozásra", type="primary"):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                new_recs, updated_recs, errors = 0, 0, 0

                for i, file in enumerate(uploaded_files):
                    status_placeholder.text(f"Feldolgozás alatt: {file.name}")
                    data = process_document_with_gemini(file)
                    if data:
                        status = upsert_record(data)
                        if status == "new": new_recs += 1
                        elif status == "update": updated_recs += 1
                        else: errors += 1
                    else:
                        errors += 1
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    if i < len(uploaded_files) - 1: time.sleep(4)

                status_placeholder.success(f"Beküldve! Sikeres: {new_recs + updated_recs} | Sikertelen: {errors}")

        st.divider()
        st.subheader("Bekerült dokumentumaim állapota")
        df_all = load_data()
        if not df_all.empty:
            # AZ ÜGYFÉL CSAK A SAJÁT ADATAIT LÁTJA!
            df_client = df_all[df_all["Feltolto_User"] == current_user]
            if not df_client.empty:
                # Elrejtjük előle a technikai oszlopokat, hogy letisztult maradjon
                st.dataframe(df_client.drop(columns=["Utolso_Modositas_Ideje", "Feltolto_User"]), use_container_width=True, hide_index=True)
            else:
                st.info("Még nem töltött fel dokumentumot.")

    # =========================================================
    # 2. ADMIN NÉZET (UX)
    # =========================================================
    elif st.session_state["role"] == "admin":
        st.title("🚗 Flotta Admin Vezérlőpult")
        st.info("Teljes körű dokumentumkezelés és Master Data hozzáférés.")

        st.subheader("1. Kézi dokumentum feldolgozás")
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
                    if i < len(uploaded_files) - 1: time.sleep(4)

                status_placeholder.success(f"Feldolgozás befejezve. Új rekord: {new_recs} | Frissített: {updated_recs} | Hiba: {errors}")

        st.divider()

        # NAPI ADATKÖZLŐ (CSAK ADMIN LÁTJA)
        st.subheader("📅 2. Napi Zárás és Adatközlő Export")
        df_admin = load_data()
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if not df_admin.empty:
            df_admin['Utolso_Modositas_Ideje'] = df_admin['Utolso_Modositas_Ideje'].fillna('')
            df_daily = df_admin[df_admin["Utolso_Modositas_Ideje"].str.startswith(today_str)]
            
            if not df_daily.empty:
                st.info(f"Ma feldolgozott tételek száma: **{len(df_daily)} db**")
                output_daily = io.BytesIO()
                with pd.ExcelWriter(output_daily, engine='openpyxl') as writer:
                    df_daily.to_excel(writer, index=False, sheet_name='Napi_Betoltes')
                
                st.download_button(
                    label=f"📥 Napi Adatközlő Letöltése (Biztosito_Betoltes_{today_str.replace('-', '')}.xlsx)",
                    data=output_daily.getvalue(),
                    file_name=f'Biztosito_Betoltes_{today_str.replace("-", "")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    type="primary"
                )
            else:
                st.info("Ma még nem történt dokumentum-feldolgozás.")

        st.divider()
        
        # MASTER DATA (CSAK ADMIN LÁTJA)
        with st.expander("🗄️ Teljes Master Data (Központi Járműnyilvántartás) megtekintése"):
            if not df_admin.empty:
                st.dataframe(df_admin, use_container_width=True, hide_index=True)
                db_output = io.BytesIO()
                with pd.ExcelWriter(db_output, engine='openpyxl') as writer:
                    df_admin.to_excel(writer, index=False, sheet_name='Master_Data')
                st.download_button("Teljes Master Data exportálása", data=db_output.getvalue(), file_name='master_data_teljes.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            else:
                st.warning("A Master Data üres.")
