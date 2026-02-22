import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import os
import io
import time
import sqlite3
import re
import bcrypt
from datetime import datetime

# =========================================================
# 1. BIZTONSÁGOS HITELÉSÍTÉSI RENDSZER (BCRYPT)
# =========================================================
def check_password():
    def password_entered():
        user = st.session_state["username"]
        pwd = st.session_state["password"]
        
        if "users" in st.secrets and user in st.secrets["users"]:
            stored_secret = st.secrets["users"][user]["password"]
            
            is_valid = False
            try:
                if bcrypt.checkpw(pwd.encode('utf-8'), stored_secret.encode('utf-8')):
                    is_valid = True
            except ValueError:
                if pwd == stored_secret:
                    is_valid = True

            if is_valid:
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
    # =========================================================
    # KONFIGURÁCIÓ ÉS VÁLTOZÓK
    # =========================================================
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        st.error("Kritikus hiba: API kulcs nem található a secrets.toml fájlban.")
        st.stop()

    DB_FILE = "fleet.db"
    
    EXPECTED_FIELDS = [
        "Dokumentum_Tipus", "Alvazszam", "Rendszam", "Vevo_Tulajdonos", 
        "Elado", "Brutto_Vetelar", "Teljesitmeny_kW", "Hengerurtartalom_cm3", "Elso_forgalomba_helyezes"
    ]
    CONF_FIELDS = [f"{f}_Conf" for f in EXPECTED_FIELDS]

    WEIGHTS = {"Alvazszam": 3, "Dokumentum_Tipus": 2, "Brutto_Vetelar": 2, "Rendszam": 1}

    # =========================================================
    # EXCEL SZÉPÍTŐ FÜGGVÉNY
    # =========================================================
    def get_formatted_excel(df, sheet_name="Adatok"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            worksheet.auto_filter.ref = worksheet.dimensions
            from openpyxl.utils import get_column_letter
            for idx, col in enumerate(df.columns, 1):
                col_letter = get_column_letter(idx)
                max_len = max(df[col].astype(str).map(len).max() if not df[col].empty else 0, len(str(col))) + 2
                worksheet.column_dimensions[col_letter].width = min(max_len, 40)
        return output.getvalue()

    # =========================================================
    # SQLITE ADATBÁZIS RÉTEG (WAL MÓD + AUDIT + TRANZAKCIÓK)
    # =========================================================
    def get_db_connection():
        conn = sqlite3.connect(DB_FILE, timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def init_db():
        conn = get_db_connection()
        cursor = conn.cursor()
        columns = [
            "Alvazszam TEXT PRIMARY KEY", "Rendszam TEXT", "Vevo_Tulajdonos TEXT", 
            "Elado TEXT", "Brutto_Vetelar TEXT", "Teljesitmeny_kW TEXT", 
            "Hengerurtartalom_cm3 TEXT", "Elso_forgalomba_helyezes TEXT",
            "Dokumentum_Tipus TEXT", "Feldolgozasi_Statusz TEXT", 
            "Utolso_Modositas_Ideje TEXT", "Feltolto_User TEXT", 
            "Hiba_Oka TEXT", "Confidence_Score REAL", "Modosito_User TEXT"
        ]
        columns.extend([f"{f} REAL" for f in CONF_FIELDS])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS masterdata ({', '.join(columns)})")
        
        try: cursor.execute("ALTER TABLE masterdata ADD COLUMN Modosito_User TEXT")
        except sqlite3.OperationalError: pass
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS masterdata_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Alvazszam TEXT,
                Field_Name TEXT,
                Old_Value TEXT,
                New_Value TEXT,
                Modified_By TEXT,
                Timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    init_db()

    def load_data():
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM masterdata", conn)
        conn.close()
        string_cols = ["Hiba_Oka", "Feldolgozasi_Statusz", "Utolso_Modositas_Ideje", "Dokumentum_Tipus", "Modosito_User"]
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        return df

    def delete_record(alvazszam):
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_user = st.session_state["logged_in_user"]
            
            # FIX 2: Atomic Transaction törlésnél
            with conn:
                cursor.execute("""
                    INSERT INTO masterdata_audit 
                    (Alvazszam, Field_Name, Old_Value, New_Value, Modified_By, Timestamp) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (alvazszam, "RECORD_STATUS", "Létező", "TÖRÖLVE", current_user, timestamp_now))
                
                cursor.execute("DELETE FROM masterdata WHERE Alvazszam=?", (alvazszam,))
            return True
        except Exception as e:
            st.error(f"Hiba a törlés során: {e}")
            return False
        finally:
            conn.close()

    def upsert_record(new_data_dict):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        alvaz = new_data_dict.get("Alvazszam")
        if not alvaz or str(alvaz).lower() in ["null", "none", ""]:
            alvaz = f"ISMERETLEN_{int(time.time())}"
            new_data_dict["Alvazszam"] = alvaz
            
        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_user = st.session_state["logged_in_user"]
        
        new_data_dict["Utolso_Modositas_Ideje"] = timestamp_now
        if "Feltolto_User" not in new_data_dict:
            new_data_dict["Feltolto_User"] = current_user

        clean_dict = {k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in new_data_dict.items() if v is not None}
        
        status = "error"
        try:
            # FIX 2: Valódi Atomi Tranzakció (with conn)
            with conn:
                cursor.execute("SELECT * FROM masterdata WHERE Alvazszam=?", (alvaz,))
                existing_row = cursor.fetchone()
                
                # FIX 1: Valós Status logika (new vs updated)
                status = "updated" if existing_row else "new"
                
                if existing_row:
                    col_names_db = [description[0] for description in cursor.description]
                    existing_dict = dict(zip(col_names_db, existing_row))
                    
                    for k, new_v in clean_dict.items():
                        old_v = existing_dict.get(k)
                        if k not in ["Utolso_Modositas_Ideje", "Feltolto_User", "Modosito_User"] and str(old_v) != str(new_v):
                            cursor.execute("""
                                INSERT INTO masterdata_audit 
                                (Alvazszam, Field_Name, Old_Value, New_Value, Modified_By, Timestamp) 
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (alvaz, k, str(old_v), str(new_v), current_user, timestamp_now))

                cols = list(clean_dict.keys())
                vals = [clean_dict[c] for c in cols]
                placeholders = ", ".join(["?"] * len(cols))
                col_names = ", ".join(cols)
                update_clause = ", ".join([f"{c} = excluded.{c}" for c in cols if c != "Alvazszam"])
                
                sql = f"INSERT INTO masterdata ({col_names}) VALUES ({placeholders}) ON CONFLICT(Alvazszam) DO UPDATE SET {update_clause}"
                cursor.execute(sql, vals)
                
        except Exception as e:
            st.error(f"Adatbázis hiba: {e}")
            status = "error"
        finally:
            conn.close()
            
        return status

    # =========================================================
    # VALIDÁCIÓS RÉTEG
    # =========================================================
    def validate_ocr_output(data):
        errors = []
        if not data: return False, "Nem valid JSON / AI hiba", 0

        weighted_sum = 0
        total_weight = 0
        for f in EXPECTED_FIELDS:
            w = WEIGHTS.get(f, 1)
            conf_val = data.get(f"{f}_Conf", 0)
            if isinstance(conf_val, (int, float)):
                weighted_sum += (conf_val * w)
                total_weight += w
        
        avg_score = (weighted_sum / total_weight) if total_weight > 0 else 0

        if data.get("low_quality_document") is True:
            errors.append("AI Jelzése: Gyenge minőségű/olvashatatlan kép!")
            avg_score -= 15

        doc_type = data.get("Dokumentum_Tipus")
        if not doc_type or str(doc_type).lower() == "null":
            errors.append("Hiányzó Dokumentum Típus"); avg_score -= 20

        alvaz = data.get("Alvazszam")
        if not alvaz or str(alvaz).lower() == "null":
            errors.append("Hiányzó Alvázszám"); avg_score -= 40; data["Alvazszam_Conf"] = 0
        else:
            clean_alvaz = str(alvaz).upper().replace(" ", "").replace("-", "")
            VIN_REGEX = r"^[A-HJ-NPR-Z0-9]{17}$"
            if not re.match(VIN_REGEX, clean_alvaz):
                errors.append(f"Érvénytelen VIN formátum ({clean_alvaz})")
                avg_score -= 40
                data["Alvazszam_Conf"] = 0

        if str(doc_type).lower() == "számla":
            vetelar = data.get("Brutto_Vetelar")
            if not vetelar or str(vetelar).lower() in ["null", "none", ""]:
                errors.append("Hiányzó Vételár (Számla)"); avg_score -= 20

        low_conf_fields = [f for f in EXPECTED_FIELDS if data.get(f"{f}_Conf", 0) < 80 and str(data.get(f, "")).lower() not in ["null", "none", ""]]
        if low_conf_fields: errors.append(f"Alacsony AI magabiztosság: {', '.join(low_conf_fields)}")

        final_score = max(0, min(100, avg_score))
        if errors: return False, " | ".join(errors), final_score
        return True, "", final_score

    # =========================================================
    # AI KINYERÉS ÉS HASH-ALAPÚ CACHE
    # =========================================================
    @st.cache_data(ttl=3600)
    def get_best_models():
        try:
            available = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            preferred = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.5-flash-latest', 'gemini-1.5-pro-latest']
            return [m for m in preferred if m in available] or (available[:1] if available else ['gemini-1.5-flash'])
        except Exception:
            return ['gemini-1.5-flash']

    # FIX 4: Költségoptimalizált AI Cache (SHA256 hash a háttérben a file_bytes alapján)
    @st.cache_data(show_spinner=False, ttl=86400) # 24 óráig emlékszik a fájlra!
    def extract_with_ai_cached(file_bytes, prompt, models_to_try):
        pdf_part = {"mime_type": "application/pdf", "data": file_bytes}
        last_error = ""
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [prompt, pdf_part],
                    generation_config=genai.GenerationConfig(response_mime_type="application/json")
                )
                
                try: raw_text = response.text
                except Exception as text_e: raise ValueError(f"AI nem adott vissza szöveget: {text_e}")

                # FIX 3: Biztonságos JSON extract
                start_idx = raw_text.find('{')
                end_idx = raw_text.rfind('}')
                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON block found in response.")
                
                clean_json_str = raw_text[start_idx : end_idx + 1]
                raw_json = json.loads(clean_json_str)
                
                flat_data = {"low_quality_document": raw_json.get("low_quality_document", False)}
                for field in EXPECTED_FIELDS:
                    if field in raw_json and isinstance(raw_json[field], dict):
                        flat_data[field] = raw_json[field].get("value")
                        if str(flat_data[field]).lower() in ["null", "none", ""]: flat_data[f"{field}_Conf"] = 0
                        else: flat_data[f"{field}_Conf"] = raw_json[field].get("confidence", 0)
                    else:
                        flat_data[field] = raw_json.get(field)
                        flat_data[f"{field}_Conf"] = 0
                        
                return flat_data, ""
            except Exception as e:
                last_error = str(e)
                continue
        return None, f"AI Rendszerhiba: {last_error}"

    def process_document_with_gemini(uploaded_file):
        models_to_try = get_best_models()
        prompt = """
        Elemezd a dokumentumot (forgalmi vagy számla) és add vissza az adatokat szigorúan JSON struktúrában!
        Minden mezőhöz kötelezően meg kell adnod egy "value" (érték) és egy "confidence" (0-100) párost.
        EXTRA: Ha a dokumentum nagyon rossz minőségű, homályos vagy nehezen olvasható, állítsd be a "low_quality_document": true értéket a gyökérszinten!
        Kinyerendő mezők: Dokumentum_Tipus, Alvazszam, Rendszam, Vevo_Tulajdonos, Elado, Brutto_Vetelar, Teljesitmeny_kW, Hengerurtartalom_cm3, Elso_forgalomba_helyezes.
        """
        # Hívjuk a CACHE-elt belső függvényt a nyers bájtokkal!
        return extract_with_ai_cached(uploaded_file.getvalue(), prompt, models_to_try)

    # =========================================================
    # OLDALSÁV ÉS PIPELINE
    # =========================================================
    with st.sidebar:
        current_user = st.session_state['logged_in_user']
        current_role = st.session_state['role']
        st.write(f"👤 Felhasználó: **{current_user}**")
        st.write(f"🛡️ Szerepkör: *{current_role.capitalize()}*")
        
        if st.sidebar.button("Kijelentkezés"):
            for key in ["password_correct", "logged_in_user", "role"]:
                if key in st.session_state: del st.session_state[key]
            st.rerun()

    def run_processing_pipeline(uploaded_files):
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        new_recs, updated_recs, validation_fails, critical_errors = 0, 0, 0, 0

        for i, file in enumerate(uploaded_files):
            status_placeholder.text(f"Státusz: OCR_Feldolgozás_Alatt - {file.name}")
            extracted_data, ai_error_message = process_document_with_gemini(file)
            
            if extracted_data:
                is_valid, error_reason, conf_score = validate_ocr_output(extracted_data)
                extracted_data["Confidence_Score"] = conf_score
                if "low_quality_document" in extracted_data: del extracted_data["low_quality_document"]
                
                if is_valid:
                    extracted_data["Feldolgozasi_Statusz"] = "Kész"; extracted_data["Hiba_Oka"] = ""
                else:
                    extracted_data["Feldolgozasi_Statusz"] = "Validáció_Szükséges"; extracted_data["Hiba_Oka"] = error_reason
                    validation_fails += 1
                
                status = upsert_record(extracted_data)
                # FIX 1: Valós status routing a pipeline-ban
                if status == "new": new_recs += 1
                elif status == "updated": updated_recs += 1
            else:
                critical_errors += 1
                upsert_record({"Dokumentum_Tipus": "Ismeretlen", "Feldolgozasi_Statusz": "Hiba", "Hiba_Oka": ai_error_message, "Confidence_Score": 0})
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            if i < len(uploaded_files) - 1: time.sleep(1)

        success_msg = f"Feldolgozás befejezve! Új bejegyzés: {new_recs} | Frissítve: {updated_recs} | Kritikus Hiba: {critical_errors}"
        if validation_fails > 0 or critical_errors > 0: st.warning(f"{success_msg} ⚠️ {validation_fails} dokumentum emberi ellenőrzést igényel!")
        else: st.success(success_msg)

    # =========================================================
    # 1. RENDSZER ADMIN NÉZET
    # =========================================================
    if st.session_state["role"] == "admin":
        st.title("🚗 IT Rendszer Admin Vezérlőpult")
        df_admin = load_data()
        
        st.subheader("🚨 AI Megbízhatósági Dashboard")
        if not df_admin.empty:
            df_errors = df_admin[df_admin["Feldolgozasi_Statusz"].isin(["Validáció_Szükséges", "Hiba"])]
            col1, col2, col3, col4 = st.columns(4)
            avg_score = df_admin["Confidence_Score"].mean() if "Confidence_Score" in df_admin.columns else 0
            
            col1.metric("Manuális Ellenőrzés Kell", len(df_errors))
            col2.metric("Átlagos Súlyozott Score", f"{avg_score:.1f}%")
            col3.metric("Összes Dokumentum", len(df_admin))
            col4.metric("AI Hibák", len(df_admin[df_admin["Feldolgozasi_Statusz"] == "Hiba"]))
            st.markdown("<br>", unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📌 Mezős Szintű Analitika", "📌 Hibás/Hiányzó Adatok", "📌 Nyers JSON Hiba", "📈 Field Failure Rate", "🕵️‍♂️ Audit Log"])
            
            with tab1:
                if not df_errors.empty:
                    disp_cols = [c for c in ["Alvazszam", "Alvazszam_Conf", "Rendszam", "Rendszam_Conf", "Brutto_Vetelar", "Brutto_Vetelar_Conf", "Hiba_Oka"] if c in df_errors.columns]
                    st.dataframe(df_errors[disp_cols].sort_values(by="Alvazszam_Conf", ascending=True), use_container_width=True, hide_index=True)
            with tab2:
                df_missing = df_errors[df_errors["Hiba_Oka"].str.contains("Hiányzó|Érvénytelen|Minőség", na=False, case=False)]
                if not df_missing.empty: st.dataframe(df_missing[["Alvazszam", "Dokumentum_Tipus", "Hiba_Oka", "Confidence_Score"]], use_container_width=True, hide_index=True)
            with tab3:
                df_json = df_errors[df_errors["Feldolgozasi_Statusz"] == "Hiba"]
                if not df_json.empty: st.dataframe(df_json[["Alvazszam", "Feldolgozasi_Statusz", "Hiba_Oka"]], use_container_width=True, hide_index=True)
            with tab4:
                st.markdown("Melyik mezőnél bizonytalanodik el a legtöbbször az AI?")
                failure_rates = []
                for f in EXPECTED_FIELDS:
                    if f"{f}_Conf" in df_admin.columns:
                        fail_rate = (df_admin[f"{f}_Conf"] < 80).mean() * 100
                        failure_rates.append({"Mező": f, "Hiba_Arány_%": round(fail_rate, 1)})
                if failure_rates:
                    df_failures = pd.DataFrame(failure_rates).sort_values("Hiba_Arány_%", ascending=False)
                    st.dataframe(df_failures, use_container_width=True, hide_index=True)
            with tab5:
                st.markdown("Ki, mikor, mit írt át vagy törölt a mesteradatban?")
                conn = get_db_connection()
                df_audit = pd.read_sql_query("SELECT * FROM masterdata_audit ORDER BY id DESC", conn)
                conn.close()
                if not df_audit.empty: st.dataframe(df_audit, use_container_width=True, hide_index=True)
                else: st.info("Még nem történt manuális adatfelülírás vagy törlés.")

        st.divider()
        st.subheader("1. Kézi dokumentum feldolgozás")
        uploaded_files = st.file_uploader("Válassza ki a PDF fájlokat", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            if st.button(f"{len(uploaded_files)} fájl feldolgozásának indítása"): run_processing_pipeline(uploaded_files)

        st.divider()
        with st.expander("🗄️ Teljes Master Data"):
            if not df_admin.empty:
                st.dataframe(df_admin, use_container_width=True, hide_index=True)
                excel_data = get_formatted_excel(df_admin, sheet_name='Master_Data')
                st.download_button(label="📥 Master Data letöltése (.xlsx)", data=excel_data, file_name='master_data_teljes.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type="primary")

    # =========================================================
    # 2. ÜZLETI ADMINISZTRÁTOR NÉZET
    # =========================================================
    elif st.session_state["role"] == "adminisztrator":
        st.title("🚗 Flotta Backoffice Vezérlőpult")
        df_admin = load_data()
        
        if not df_admin.empty:
            df_pending = df_admin[df_admin["Feldolgozasi_Statusz"].isin(["Validáció_Szükséges", "Hiba"])].copy()
            if not df_pending.empty:
                st.error(f"⚠️ {len(df_pending)} tétel manuális javításra vagy törlésre szorul!")
                st.info("Kattints duplán a cellákra a javításhoz, majd pipáld be a **Jóváhagyás** oszlopot a mentéshez! Hibás feltöltés esetén használd a **Törlés** oszlopot.")
                
                cols_to_drop = [c for c in df_pending.columns if c.endswith("_Conf") or c in ["Confidence_Score", "Feltolto_User", "Utolso_Modositas_Ideje", "Modosito_User"]]
                df_editable = df_pending.drop(columns=cols_to_drop, errors='ignore')
                
                df_editable.insert(0, "Jóváhagyás", False)
                df_editable.insert(1, "Törlés", False)
                
                edited_df = st.data_editor(df_editable, hide_index=True, use_container_width=True, key="data_editor")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Kijelölt sorok Mentése", type="primary"):
                        approved_rows = edited_df[(edited_df["Jóváhagyás"] == True) & (edited_df["Törlés"] == False)]
                        if not approved_rows.empty:
                            for _, row in approved_rows.iterrows():
                                r_dict = row.to_dict()
                                del r_dict["Jóváhagyás"]
                                del r_dict["Törlés"]
                                r_dict["Feldolgozasi_Statusz"] = "Kész"
                                r_dict["Hiba_Oka"] = "Kézi javítás"
                                r_dict["Modosito_User"] = current_user
                                upsert_record(r_dict)
                            st.success(f"Sikeresen frissítve {len(approved_rows)} tétel!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Nincs kijelölve egyetlen sor sem a jóváhagyáshoz.")

                with col2:
                    if st.button("🗑️ Kijelölt sorok Törlése", type="secondary"):
                        to_delete_rows = edited_df[edited_df["Törlés"] == True]
                        if not to_delete_rows.empty:
                            for _, row in to_delete_rows.iterrows():
                                delete_record(row["Alvazszam"])
                            st.success(f"Sikeresen törölve {len(to_delete_rows)} tétel az adatbázisból!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.warning("Nincs kijelölve egyetlen sor sem a törléshez.")

        st.divider()
        st.subheader("1. Kézi dokumentum feldolgozás")
        uploaded_files = st.file_uploader("Válassza ki a PDF fájlokat", type=['pdf'], accept_multiple_files=True, key="bo_uploader")
        if uploaded_files:
            if st.button(f"{len(uploaded_files)} fájl feldolgozásának indítása", key="bo_btn"): run_processing_pipeline(uploaded_files)

        st.divider()
        st.subheader("📅 2. Napi Zárás és Adatközlő Export")
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        if not df_admin.empty:
            df_daily = df_admin[(df_admin["Utolso_Modositas_Ideje"].str.startswith(today_str)) & (df_admin["Feldolgozasi_Statusz"] == "Kész")]
            cols_to_drop = [c for c in df_daily.columns if c.endswith("_Conf") or c == "Confidence_Score"]
            df_daily_clean = df_daily.drop(columns=cols_to_drop, errors='ignore')
            
            if not df_daily_clean.empty:
                st.success(f"Ma feldolgozott, KÉSZ tételek száma: **{len(df_daily_clean)} db**")
                daily_excel = get_formatted_excel(df_daily_clean, sheet_name='Napi_Betoltes')
                st.download_button(label=f"📥 Napi Adatközlő Letöltése (.xlsx)", data=daily_excel, file_name=f'Biztosito_Betoltes_{today_str.replace("-", "")}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type="primary")
            else:
                st.info("Ma még nem történt sikeres dokumentum-feldolgozás.")
                
        st.divider()
        with st.expander("🗄️ Tiszta Master Data"):
            if not df_admin.empty:
                cols_to_drop = [c for c in df_admin.columns if c.endswith("_Conf") or c == "Confidence_Score"]
                df_admin_clean = df_admin.drop(columns=cols_to_drop, errors='ignore')
                st.dataframe(df_admin_clean, use_container_width=True, hide_index=True)
                clean_excel = get_formatted_excel(df_admin_clean, sheet_name='Master_Data')
                st.download_button(label="📥 Tiszta Master Data letöltése (.xlsx)", data=clean_excel, file_name='master_data_tiszta.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', type="secondary")

    # =========================================================
    # 3. ÜGYFÉL NÉZET
    # =========================================================
    elif st.session_state["role"] == "ugyfel":
        st.title("📁 Dokumentum Feltöltő Központ")
        
        uploaded_files = st.file_uploader("PDF fájlok kiválasztása", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            if st.button(f"{len(uploaded_files)} fájl beküldése feldolgozásra", type="primary"): run_processing_pipeline(uploaded_files)

        st.divider()
        st.subheader("Bekerült dokumentumaim állapota")
        df_all = load_data()
        if not df_all.empty:
            df_client = df_all[df_all["Feltolto_User"] == current_user]
            if not df_client.empty:
                display_cols = ["Alvazszam", "Dokumentum_Tipus", "Feldolgozasi_Statusz", "Hiba_Oka"]
                st.dataframe(df_client[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("Még nem töltött fel dokumentumot.")
