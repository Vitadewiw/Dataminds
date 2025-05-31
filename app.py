
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
def load_data():
    df = pd.read_csv("HR_FINPRO.csv")  # Sesuaikan dengan path dataset Anda
    return df

@st.cache_resource
def load_model():
    return joblib.load('rf_top_10.joblib')

model = load_model()


df = load_data()

# Mapping label asli dari data yang telah dienkode
label_mapping = {
    "Attrition": {0: "No", 1: "Yes"},
}

# Ubah kembali nilai yang sudah dienkode menjadi label asli
for column, mapping in label_mapping.items():
    if column in df.columns:
        df[column] = df[column].map(mapping)

# Sidebar
st.sidebar.title("HR Attrition Prediction")
menu = st.sidebar.radio("Pilih Menu:", ["Home", "Prediction", "Batch Prediction"])


st.title("HR Attrition Prediction")

# =============================== HOME ===============================
if menu == "Home":
    st.subheader("Dashboard Karyawan Berdasarkan Attrition")
    attrition_status = st.selectbox("Pilih Status Attrition:", df["Attrition"].unique())
    filtered_df = df[df["Attrition"] == attrition_status]

    # Menampilkan diagram Top 10 Feature berdasarkan Feature Importance
    if model is not None and hasattr(model, "feature_importances_"):
        st.write("### 🔍 Top 10 Fitur Berdasarkan Importance")
        try:
            # Harus sesuai urutan input ke model
            feature_names = ['MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome', 
                             'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction', 
                             'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction' 
            ]
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                "Fitur": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            # Plot
            plt.figure(figsize=(10, 5))
            sns.barplot(x="Importance", y="Fitur", data=feature_df, palette="viridis")
            plt.title("Top 10 Fitur Berdasarkan Importance")
            plt.tight_layout()
            st.pyplot(plt)

            # Tabel
            st.write("### 📋 Tabel Top 10 Fitur")
            st.dataframe(feature_df.reset_index(drop=True))

        except Exception as e:
            st.error(f"Gagal memuat feature importance: {e}")
    else:
        st.warning("Model belum dimuat atau tidak memiliki atribut feature_importances_.")


# ============================ PREDICTION ============================
elif menu == "Prediction":
    if model is None:
        st.error("Model not loaded.")
    else:
        st.title('Employee Attrition Prediction')
        st.markdown("""
        <div style='background-color: black; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
            <span style='color:white'><b>This app predicts employee attrition based on key work-related factors 🚀. Enter the details and get an instant prediction!</b></span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Enter Employee Details")

        with st.form("form_prediksi"):
            col1, col2 = st.columns(2)

            with col1:
                MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
                MaritalStatus_Single = 1 if MaritalStatus == "Single" else 0
                JobLevelSatisfaction = st.selectbox("Job Level Satisfaction (1–4)", [1, 2, 3, 4])
                MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
                StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])
                JobInvolvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])

            with col2:
                EmployeeSatisfaction = st.selectbox("Employee Satisfaction (1–4)", [1, 2, 3, 4])
                DailyRate = st.number_input("Daily Rate (opsional)", min_value=0, max_value=1500, value=0)
                DistanceFromHome = st.number_input("Distance From Home (km)", 0, 100, 10)
                Age = st.number_input("Age", min_value=18, max_value=60, value=30)
                EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1–4)", [1, 2, 3, 4])

            submit = st.form_submit_button("Predict")

        if submit:
            input_data = pd.DataFrame([[  # Input untuk prediksi
                MaritalStatus_Single,
                JobLevelSatisfaction,
                MonthlyIncome,
                StockOptionLevel,
                JobInvolvement,
                EmployeeSatisfaction,
                DailyRate,
                DistanceFromHome,
                Age,
                EnvironmentSatisfaction
            ]], columns=[
                'MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome', 
                'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction', 
                'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction'
            ])

            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(input_data)[0]
                prob_resign = round(probas[1] * 100, 2)
                prob_stay = round(probas[0] * 100, 2)

                if prediction == 1:
                    st.info(f"📊 Peluang karyawan untuk **resign**: **{prob_resign}%**")
                else:
                    st.info(f"📊 Peluang karyawan untuk **bertahan**: **{prob_stay}%**")
            else:
                st.warning("📊 Model tidak mendukung prediksi probabilitas.")

            # Tampilkan hasil utama
            result = "✅ Karyawan kemungkinan bertahan." if prediction == 0 else "⚠️ Karyawan kemungkinan resign."
            st.success(f"Hasil Prediksi: {result}")

            # Rekomendasi berdasarkan hasil prediksi
            if prediction == 0:
                st.markdown("""
                ### ✅ **📈 Rekomendasi Tindakan Jika Karyawan Bertahan**
                1. **Optimalkan Gaji dan Tunjangan:**  
                   Lakukan peninjauan terhadap struktur kompensasi agar tetap kompetitif dan sesuai dengan kontribusi karyawan.

                2. **Fokus pada Pengembangan Karier:**  
                   Berikan peluang pengembangan diri seperti pelatihan, sertifikasi, dan mentoring yang mendukung jenjang karier.

                3. **Peningkatan Keseimbangan Kerja-Hidup:**  
                   Evaluasi ulang kebijakan jam kerja fleksibel dan opsi kerja jarak jauh (remote/hybrid).

                4. **Bangun Keterlibatan Jangka Panjang:**  
                   Lakukan survei kepuasan karyawan secara berkala dan tindak lanjuti hasilnya melalui diskusi terbuka dan program peningkatan.
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                ### ❌ **📉 Rekomendasi Tindakan Jika Karyawan Resign**
                1. **Lakukan Exit Interview:**  
                   Kumpulkan data terkait alasan pengunduran diri untuk mendapatkan wawasan yang jujur dan bermanfaat.

                2. **Analisis Pola Turnover:**  
                   Gunakan data historis untuk mengidentifikasi tren seperti unit kerja dengan turnover tinggi atau waktu kerja rata-rata sebelum resign.

                3. **Perbaiki Lingkungan Kerja:**  
                   Tindak lanjuti temuan dari exit interview dalam bentuk perbaikan manajemen, beban kerja, atau budaya organisasi.

                4. **Strategi Retensi Masa Depan:**  
                   Bangun program retensi berbasis data yang menargetkan karyawan dengan potensi tinggi dan risiko tinggi keluar.
                """, unsafe_allow_html=True)


# =============================== Batch Prediction ===============================
elif menu == "Batch Prediction":
    st.subheader("📂 Upload File Karyawan untuk Prediksi Massal")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            # Validasi apakah kolom sesuai dengan yang diharapkan oleh model
            expected_columns = [
                'MaritalStatus_Single', 'JobLevelSatisfaction', 'MonthlyIncome',
                'StockOptionLevel', 'JobInvolvement', 'EmployeeSatisfaction',
                'DailyRate', 'DistanceFromHome', 'Age', 'EnvironmentSatisfaction'
            ]

            if not all(col in batch_df.columns for col in expected_columns):
                st.error("⚠️ File CSV harus memiliki kolom berikut:\n" + ", ".join(expected_columns))
            else:
                # Prediksi
                predictions = model.predict(batch_df)
                probabilities = model.predict_proba(batch_df)

                batch_df['Prediksi'] = np.where(predictions == 1, 'Resign', 'Bertahan')
                batch_df['Peluang Resign (%)'] = (probabilities[:, 1] * 100).round(2)
                batch_df['Peluang Bertahan (%)'] = (probabilities[:, 0] * 100).round(2)

                st.success("✅ Prediksi berhasil dilakukan!")
                st.markdown("### 📊 Hasil Prediksi Karyawan:")
                st.dataframe(batch_df)

                # Download hasil
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Hasil Prediksi sebagai CSV",
                    data=csv,
                    file_name="hasil_prediksi_karyawan.csv",
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")
