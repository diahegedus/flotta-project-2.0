import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import io
import time
from datetime import datetime

# --- 1. HITELÉSÍTÉSI RENDSZER ---
def check_password():
    def password_entered():
        user = st.session_state["username"]
        pwd = st.session_state["password"]
        
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
    
    # Új oszlopok a mezőnkénti confidence score-okhoz
    EXPECTED_FIELDS = [
        "Dokumentum_Tipus", "Alvazszam", "Rendszam", "Vevo_Tulajdonos", 
        "Elado", "Brutto_Vetelar", "Teljesitmeny_kW", "Hengerurtartalom_cm3", "Elso_forgalomba_helyezes"
    ]
    CONF_FIELDS = [f"{f}_Conf" for f in EXPECTED_FIELDS]

    # --- MASTER DATA KEZELÉS ---
    def load_data():
        if os.path.exists(DB_FILE):
            df = pd.read_csv(DB_FILE)
            if "Feltolto_User" not in df.columns: df["Feltolto_User"] = "ismeretlen"
            if "Hiba_Oka" not in df.columns: df["Hiba_Oka"] = ""
            if "Confidence_Score" not in df.columns: df["Confidence_Score"] = 100
            for conf_f in CONF_FIELDS:
                if conf_f not in df.columns: df[conf_f] = 0
            return df
        
        cols = EXPECTED_FIELDS + CONF_FIELDS + [
            "Feldolgozasi_Statusz", "Utolso_Modositas_Ideje", 
            "Feltolto_User", "Hiba_Oka", "Confidence_Score"
        ]
        return pd.DataFrame(columns=cols)

    def save_data(df):
        df.to_csv(DB_FILE, index=False)

    def upsert_record(new_data_dict):
        df = load_data()
        alvaz = new_data_dict.get("Alvazszam")
        
        new_data_dict["Utolso_Modositas_Ideje"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_data_dict["Feltolto_User"] = st.session_state["logged_in_user"]

        if alvaz and str(alvaz).lower() not in ["null", "none", ""]:
            if alvaz in df["Alvazszam"].values:
                idx = df.index[df['Alvazszam'] == alvaz][0]
                for key, value in new_data_dict.items():
                    if value is not None and str(value).lower() not in ["null", "none", ""]: 
                        df.at[idx, key] = value
                save_data(df)
                return "update"
            else:
                new_row = pd.DataFrame([new_data_dict])
                df = pd.concat([df, new_row], ignore_index=True)
                save_data(df)
                return "new"
        
        new_data_dict["Alvazszam"] = f"ISMERETLEN_{int(time.time())}"
        new_row = pd.DataFrame([new_data_dict])
        df = pd.concat([df, new_row], ignore_index=True)
        save_data(df)
        return "new"

    # --- 2. VALIDÁCIÓS RÉTEG (HIBRID: AI Score + Szabályok) ---
    def validate_ocr_output(data):
        errors = []
        
        if not data:
            return False, "Nem valid JSON / AI hiba", 0

        # Átlagos AI confidence számítása a mezőkből
        total_conf = 0
        valid_fields = 0
        for f in EXPECTED_FIELDS:
            conf_val = data.get(f"{f}_Conf", 0)
            if isinstance(conf_val, (int, float)):
                total_conf += conf_val
                valid_fields += 1
        
        avg_score = (total_conf / valid_fields) if valid_fields > 0 else 0

        doc_type = data.get("Dokumentum_Tipus")
        if not doc_type or str(doc_type).lower() == "null":
            errors.append("Hiányzó Dokumentum Típus")
            avg_score -= 20

        alvaz = data.get("Alvazszam")
        if not alvaz or str(alvaz).lower() == "null":
            errors.append("Hiányzó Alvázszám")
            avg_score -= 40
            data["Alvazszam_Conf"] = 0 # Biztosan rossz
        else:
            clean_alvaz = str(alvaz).replace(" ", "").replace("-", "")
            if len(clean_alvaz) != 17:
                errors.append(f"Érvénytelen VIN hossz ({len(clean_alvaz)} kar.)")
                avg_score -= 40
                data["Alvazszam_Conf"] = 0 # Szabály felülírja az AI magabiztosságát

        if str(doc_type).lower() == "számla":
            vetelar = data.get("Brutto_Vetelar")
            if not vetelar or str(vetelar).lower() in ["null", "none", ""]:
                errors.append("Hiányzó Vételár (Számla)")
                avg_score -= 20

        # Ha bármelyik mező magabiztossága 80% alatti, küldjük ellenőrzésre!
        low_conf_fields = [f for f in EXPECTED_FIELDS if data.get(f"{f}_Conf", 0) < 80 and str(data.get(f, "")).lower() not in ["null", "none", ""]]
        if low_conf_fields:
            errors.append(f"Alacsony AI magabiztosság: {', '.join(low_conf_fields)}")

        final_score = max(0, min(100, avg_score)) # 0-100 között tartjuk

        if errors:
            return False, " | ".join(errors), final_score
        return True, "", final_score

    # --- AI KINYERÉS ÉS JSON LAPÍTÁS ---
    def process_document_with_gemini(uploaded_file):
        try:
            available_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except:
            available_models = []
            
        preferred_order = ['gemini-1.5-flash', 'gemini-1.5-pro']
        models_to_try = [m for m in preferred_order if m in available_models] or (available_models[:1] if available_models else [])

        # VÁLTOZÁS: Nested JSON formátumot kérünk value és confidence párosokkal!
        prompt = """
        Elemezd a dokumentumot (forgalmi vagy számla) és add vissza az adatokat szigorúan az alábbi JSON struktúrában!
        Minden mezőhöz kötelezően meg kell adnod egy "value" (érték) és egy "confidence" (0-100 közötti magabiztossági százalék) párost.
        
        Példa formátum:
        {
          "Dokumentum_Tipus": {"value": "Számla", "confidence": 100},
          "Alvazszam": {"value": "WBA1234567890ABCD", "confidence": 95}
        }
        
        Kinyerendő mezők:
        - Dokumentum_Tipus ("Forgalmi" vagy "Számla")
        - Alvazszam (17 karakteres VIN)
        - Rendszam 
        - Vevo_Tulajdonos (Vevő vagy C.1 kód)
        - Elado (Csak számla esetén)
        - Brutto_Vetelar (Csak számla esetén, számérték)
        - Teljesitmeny_kW (P.2 kód)
        - Hengerurtartalom_cm3 (P.1 kód)
        - Elso_forgalomba_helyezes (B kód)

        Csak a nyers JSON-t add vissza!
        """
        pdf_part = {"mime_type": "application/pdf", "data": uploaded_file.getvalue()}
        
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content([prompt, pdf_part])
                clean_text = response.text.replace('```json', '').replace('```', '').strip()
                raw_json = json.loads(clean_text)
                
                # JSON LAPÍTÁSA (Flattening) az adatbázishoz
                flat_data = {}
                for field in EXPECTED_FIELDS:
                    if field in raw_json and isinstance(raw_json[field], dict):
                        flat_data[field] = raw_json[field].get("value")
                        # Ha az AI null-t ad vissza, a confidence legyen 0
                        if str(flat_data[field]).lower() in ["null", "none", ""]:
                            flat_data[f"{field}_Conf"] = 0
                        else:
                            flat_data[f"{field}_Conf"] = raw_json[field].get("confidence", 0)
                    else:
                        flat_data[field] = raw_json.get(field)
                        flat_data[f"{field}_Conf"] = 0
                
                return flat_data
            except Exception as e:
                continue
        return None

    # --- OLDALSÁV ---
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
    # KÖZÖS FELDOLGOZÓ LOGIKA
    # =========================================================
    def run_processing_pipeline(uploaded_files):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        new_recs, updated_recs, validation_fails, critical_errors = 0, 0, 0, 0

        for i, file in enumerate(uploaded_files):
            status_placeholder.text(f"Státusz: AI Kinyerés és Pontozás - {file.name}")
            extracted_data = process_document_with_gemini(file)
            
            if extracted_data:
                is_valid, error_reason, conf_score = validate_ocr_output(extracted_data)
                
                extracted_data["Confidence_Score"] = conf_score
                
                if is_valid:
                    extracted_data["Feldolgozasi_Statusz"] = "Kész"
                    extracted_data["Hiba_Oka"] = ""
                else:
                    extracted_data["Feldolgozasi_Statusz"] = "Validáció_Szükséges"
                    extracted_data["Hiba_Oka"] = error_reason
                    validation_fails += 1
                
                status = upsert_record(extracted_data)
                if status == "new": new_recs += 1
                elif status == "update": updated_recs += 1
            else:
                critical_errors += 1
                error_data = {
                    "Dokumentum_Tipus": "Ismeretlen",
                    "Feldolgozasi_Statusz": "Hiba",
                    "Hiba_Oka": "Nem valid JSON / AI hiba",
                    "Confidence_Score": 0
                }
                upsert_record(error_data)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            if i < len(uploaded_files) - 1: time.sleep(4)

        success_msg = f"Feldolgozás befejezve! Új: {new_recs} | Frissített: {updated_recs} | Hiba: {critical_errors}"
        if validation_fails > 0 or critical_errors > 0:
            st.warning(f"{success_msg} ⚠️ {validation_fails} dokumentum emberi ellenőrzést igényel az alacsony megbízhatóság miatt!")
        else:
            st.success(success_msg)


    # =========================================================
    # ADMIN NÉZET
    # =========================================================
    if st.session_state["role"] == "admin":
        st.title("🚗 Flotta Admin Vezérlőpult")
        
        df_admin = load_data()
        
        # --- HIBAKEZELÉSI DASHBOARD ---
        st.subheader("🚨 AI Megbízhatósági Dashboard (Field-Level Confidence)")
        if not df_admin.empty:
            df_errors = df_admin[df_admin["Feldolgozasi_Statusz"].isin(["Validáció_Szükséges", "Hiba"])]
            
            col1, col2, col3, col4 = st.columns(4)
            avg_score = df_admin["Confidence_Score"].mean() if "Confidence_Score" in df_admin.columns else 0
            
            col1.metric("Manuális Ellenőrzés Kell", len(df_errors))
            col2.metric("Átlagos Rendszer Score", f"{avg_score:.1f}%")
            col3.metric("Összes Dokumentum", len(df_admin))
            col4.metric("AI Hibák", len(df_admin[df_admin["Feldolgozasi_Statusz"] == "Hiba"]))

            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3 = st.tabs(["📌 Mezős Szintű Analitika (Alacsony Pontszámok)", "📌 Hibás/Hiányzó Adatok", "📌 Nyers JSON Összeomlások"])
            
            with tab1:
                st.markdown("Az AI az alábbi dokumentumoknál **bizonyos mezőkben bizonytalan** (<80%), ezért ellenőrzésre küldte őket.")
                if not df_errors.empty:
                    # Kigyűjtjük a legfontosabb mezőket és azok pontszámait megjelenítésre
                    disp_cols = ["Alvazszam", "Alvazszam_Conf", "Rendszam", "Rendszam_Conf", "Brutto_Vetelar", "Brutto_Vetelar_Conf", "Hiba_Oka"]
                    # Csak azokat az oszlopokat mutatjuk, amik tényleg léteznek a df-ben
                    disp_cols = [c for c in disp_cols if c in df_errors.columns]
                    st.dataframe(df_errors[disp_cols].sort_values(by="Alvazszam_Conf", ascending=True), use_container_width=True, hide_index=True)
            
            with tab2:
                df_missing = df_errors[df_errors["Hiba_Oka"].str.contains("Hiányzó|Érvénytelen", na=False, case=False)]
                if not df_missing.empty:
                    st.dataframe(df_missing[["Alvazszam", "Dokumentum_Tipus", "Hiba_Oka", "Confidence_Score"]], use_container_width=True, hide_index=True)
            
            with tab3:
                df_json = df_errors[df_errors["Feldolgozasi_Statusz"] == "Hiba"]
                if not df_json.empty:
                    st.dataframe(df_json[["Alvazszam", "Feldolgozasi_Statusz", "Hiba_Oka"]], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("1. Kézi dokumentum feldolgozás")
        uploaded_files = st.file_uploader("Válassza ki a PDF fájlokat", type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            if st.button(f"{len(uploaded_files)} fájl feldolgozásának indítása"):
                run_processing_pipeline(uploaded_files)

        st.divider()
        with st.expander("🗄️ Teljes Master Data (AI Magabiztossági Pontokkal)"):
            if not df_admin.empty:
                st.dataframe(df_admin, use_container_width=True, hide_index=True)
                db_output = io.BytesIO()
                with pd.ExcelWriter(db_output, engine='openpyxl') as writer:
                    df_admin.to_excel(writer, index=False, sheet_name='Master_Data')
                st.download_button("Exportálás Excelbe", data=db_output.getvalue(), file_name='master_data_teljes.xlsx')

    # =========================================================
    # ÜGYFÉL NÉZET
    # =========================================================
    elif st.session_state["role"] == "ugyfel":
        st.title("📁 Dokumentum Feltöltő Központ")
        
        uploaded_files = st.file_uploader("PDF fájlok kiválasztása", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            if st.button(f"{len(uploaded_files)} fájl beküldése feldolgozásra", type="primary"):
                run_processing_pipeline(uploaded_files)

        st.divider()
        st.subheader("Bekerült dokumentumaim állapota")
        df_all = load_data()
        if not df_all.empty:
            df_client = df_all[df_all["Feltolto_User"] == current_user]
            if not df_client.empty:
                display_cols = ["Alvazszam", "Dokumentum_Tipus", "Feldolgozasi_Statusz"]
                st.dataframe(df_client[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Még nem töltött fel dokumentumot.")
