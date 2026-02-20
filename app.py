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
    # --- BEÁLLÍTÁSOK ---
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        st.error("❌ API kulcs hiányzik a Secrets-ből!")
        st.stop()

    DB_FILE = "forgalmi_adatbazis.csv"

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
            Elemezd a PDF-et (forgalmi vagy számla). Keresd meg:
            - Dokumentum_Tipus: "Forgalmi" vagy "Számla"
            - Alvazszam: 17 karakteres VIN (Kritikus!)
            - Rendszam: Ha van
            - Vevo_Tulajdonos: Vevő neve vagy C.1 kód alatti név
            - Elado: Csak számla esetén
            - Brutto_Vetelar: Csak számla esetén (csak a számérték)
            - Teljesitmeny_kW (P.2), Hengerurtartalom_cm3 (P.1), Elso_forgalomba_helyezes (B)

            Csak nyers JSON-t adj vissza!
            """
            pdf_part = {"mime_type": "application/pdf", "data": uploaded_file.getvalue()}
            response = model.generate_content([prompt, pdf_part])
            clean_text = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except:
            return None

    # --- FELÜLET ---
    st.title("🚗 Flotta Admin: Tömeges Adatkinyerő")
    
    with st.sidebar:
        st.write(f"👤 Felhasználó: {st.secrets['credentials']['username']}")
        if st.button("Kijelentkezés"):
            if "password_correct" in st.session_state:
                del st.session_state["password_correct"]
            st.rerun()

    # TÖMEGES FELTÖLTÉS
    uploaded_files = st.file_uploader("Dokumentumok feltöltése (PDF)", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f"{len(uploaded_files)} fájl feldolgozása", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            new_count, update_count, error_count = 0, 0, 0

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Feldolgozás: {file.name}")
                result_data = process_document_with_gemini(file)
                
                if result_data:
                    res = upsert_record(result_data)
                    if res == "new": new_count += 1
                    elif res == "update": update_count += 1
                    else: error_count += 1
                else:
                    error_count += 1
                
                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.success(f"Kész! ✨ Új: {new_count} | Frissítve: {update_count} | Hiba: {error_count}")
            st.balloons()

    st.divider()
    
    st.subheader("📊 Központi Járműnyilvántartás")
    df_admin = load_data()
    
    if not df_admin.empty:
        # Megjelenítés
        st.dataframe(df_admin, use_container_width=True, hide_index=True)
        
        # Excel export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_admin.to_excel(writer, index=False, sheet_name='Flotta_Lista')
        
        st.download_button(
            label="📥 Teljes adatbázis letöltése (.xlsx)",
            data=output.getvalue(),
            file_name='flotta_nyilvantartas.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info("Az adatbázis jelenleg üres.")
