from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook

from datetime import datetime
import json
import pandas as pd


# =========================================================
# FILE RAW
# =========================================================

WEATHER_FILE = "/usr/local/airflow/data/raw/weather_raw.json"
AIR_QUALITY_FILE = "/usr/local/airflow/data/raw/air_quality_raw.json"
MARINE_FILE = "/usr/local/airflow/data/raw/marine_raw.json"


# =========================================================
# DATABASE
# =========================================================

MYSQL_CONN_ID = "mysql_weather"


# =========================================================
# HELPER
# =========================================================

def safe_value(value):
    """
    Mengubah NaN/NaT menjadi None agar dapat disimpan sebagai NULL di MySQL.
    Nilai asli tetap dipertahankan jika tersedia.
    """
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


# =========================================================
# FUNGSI ETL
# =========================================================

def extract_transform_load():

    print("========================================")
    print("MEMULAI ETL WEATHER ML")
    print("========================================")


    # =====================================================
    # 1. EXTRACT WEATHER
    # =====================================================

    print("\n=== EXTRACT WEATHER DATA ===")

    with open(WEATHER_FILE, "r", encoding="utf-8") as f:
        weather = json.load(f)

    weather_df = pd.DataFrame(weather["hourly"])

    weather_df = weather_df[
        [
            "time",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "cloud_cover",
        ]
    ]

    print(f"Jumlah weather data: {len(weather_df)} baris")
    print("Kolom weather:")
    print(weather_df.columns.tolist())


    # =====================================================
    # 2. EXTRACT AIR QUALITY
    # =====================================================

    print("\n=== EXTRACT AIR QUALITY DATA ===")

    with open(AIR_QUALITY_FILE, "r", encoding="utf-8") as f:
        air_quality = json.load(f)

    air_df = pd.DataFrame(air_quality["hourly"])

    air_df = air_df[
        [
            "time",
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone",
        ]
    ]

    print(f"Jumlah air quality data: {len(air_df)} baris")
    print("Kolom air quality:")
    print(air_df.columns.tolist())


    # =====================================================
    # 3. EXTRACT MARINE
    # =====================================================

    print("\n=== EXTRACT MARINE DATA ===")

    with open(MARINE_FILE, "r", encoding="utf-8") as f:
        marine = json.load(f)

    marine_df = pd.DataFrame(marine["hourly"])

    marine_df = marine_df[
        [
            "time",
            "wave_height",
            "wave_period",
            "sea_surface_temperature",
        ]
    ]

    print(f"Jumlah marine data: {len(marine_df)} baris")
    print("Kolom marine:")
    print(marine_df.columns.tolist())


    # =====================================================
    # 4. TRANSFORM DASAR
    # =====================================================
    # Hanya memperbaiki tipe kolom waktu.
    # Missing value, duplicate, dan cleaning lainnya
    # sengaja TIDAK dilakukan di Airflow.
    #
    # Cleaning akan dilakukan nanti di Google Colab
    # sesuai struktur tugas.
    # =====================================================

    print("\n=== TRANSFORM DASAR ===")

    weather_df["time"] = pd.to_datetime(
        weather_df["time"],
        errors="coerce"
    )

    air_df["time"] = pd.to_datetime(
        air_df["time"],
        errors="coerce"
    )

    marine_df["time"] = pd.to_datetime(
        marine_df["time"],
        errors="coerce"
    )

    print("Konversi tipe data time selesai.")


    # =====================================================
    # 5. CEK AWAL DATA
    # =====================================================

    print("\n=== INFORMASI AWAL DATA ===")

    print("\nMissing value WEATHER:")
    print(weather_df.isnull().sum())

    print("\nMissing value AIR QUALITY:")
    print(air_df.isnull().sum())

    print("\nMissing value MARINE:")
    print(marine_df.isnull().sum())

    print("\nDuplicate WEATHER:")
    print(weather_df.duplicated().sum())

    print("\nDuplicate AIR QUALITY:")
    print(air_df.duplicated().sum())

    print("\nDuplicate MARINE:")
    print(marine_df.duplicated().sum())


    # =====================================================
    # 6. CONNECT MYSQL
    # =====================================================

    print("\n=== CONNECT DATABASE ===")

    hook = MySqlHook(
        mysql_conn_id=MYSQL_CONN_ID
    )

    print("Berhasil membuat koneksi ke MySQL")


    # =====================================================
    # 7. BUAT ULANG TABEL
    # =====================================================
    # Tabel dibuat ulang supaya setiap tabel hanya berisi
    # variabel dari dataset masing-masing.
    # =====================================================

    print("\n=== MEMBUAT TABEL DATABASE ===")


    # -----------------------------------------------------
    # WEATHER DATA
    # -----------------------------------------------------

    hook.run("""
        DROP TABLE IF EXISTS weather_data
    """)

    hook.run("""
        CREATE TABLE weather_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time DATETIME NULL,
            temperature_2m DOUBLE NULL,
            relative_humidity_2m DOUBLE NULL,
            precipitation DOUBLE NULL,
            wind_speed_10m DOUBLE NULL,
            surface_pressure DOUBLE NULL,
            cloud_cover DOUBLE NULL
        )
    """)


    # -----------------------------------------------------
    # AIR QUALITY DATA
    # -----------------------------------------------------

    hook.run("""
        DROP TABLE IF EXISTS air_quality_data
    """)

    hook.run("""
        CREATE TABLE air_quality_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time DATETIME NULL,
            pm2_5 DOUBLE NULL,
            pm10 DOUBLE NULL,
            nitrogen_dioxide DOUBLE NULL,
            ozone DOUBLE NULL
        )
    """)


    # -----------------------------------------------------
    # MARINE DATA
    # -----------------------------------------------------

    hook.run("""
        DROP TABLE IF EXISTS marine_data
    """)

    hook.run("""
        CREATE TABLE marine_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            time DATETIME NULL,
            wave_height DOUBLE NULL,
            wave_period DOUBLE NULL,
            sea_surface_temperature DOUBLE NULL
        )
    """)

    print("Tiga tabel berhasil dibuat ulang")


    # =====================================================
    # 8. LOAD WEATHER DATA
    # =====================================================

    print("\n=== LOAD WEATHER DATA ===")

    weather_rows = []

    for _, row in weather_df.iterrows():

        time_value = (
            row["time"].to_pydatetime()
            if not pd.isna(row["time"])
            else None
        )

        weather_rows.append(
            (
                time_value,
                safe_value(row["temperature_2m"]),
                safe_value(row["relative_humidity_2m"]),
                safe_value(row["precipitation"]),
                safe_value(row["wind_speed_10m"]),
                safe_value(row["surface_pressure"]),
                safe_value(row["cloud_cover"]),
            )
        )

    hook.insert_rows(
        table="weather_data",
        rows=weather_rows,
        target_fields=[
            "time",
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "cloud_cover",
        ],
    )

    print(
        f"Weather data berhasil masuk: "
        f"{len(weather_rows)} baris"
    )


    # =====================================================
    # 9. LOAD AIR QUALITY DATA
    # =====================================================

    print("\n=== LOAD AIR QUALITY DATA ===")

    air_rows = []

    for _, row in air_df.iterrows():

        time_value = (
            row["time"].to_pydatetime()
            if not pd.isna(row["time"])
            else None
        )

        air_rows.append(
            (
                time_value,
                safe_value(row["pm2_5"]),
                safe_value(row["pm10"]),
                safe_value(row["nitrogen_dioxide"]),
                safe_value(row["ozone"]),
            )
        )

    hook.insert_rows(
        table="air_quality_data",
        rows=air_rows,
        target_fields=[
            "time",
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "ozone",
        ],
    )

    print(
        f"Air quality data berhasil masuk: "
        f"{len(air_rows)} baris"
    )


    # =====================================================
    # 10. LOAD MARINE DATA
    # =====================================================

    print("\n=== LOAD MARINE DATA ===")

    marine_rows = []

    for _, row in marine_df.iterrows():

        time_value = (
            row["time"].to_pydatetime()
            if not pd.isna(row["time"])
            else None
        )

        marine_rows.append(
            (
                time_value,
                safe_value(row["wave_height"]),
                safe_value(row["wave_period"]),
                safe_value(row["sea_surface_temperature"]),
            )
        )

    hook.insert_rows(
        table="marine_data",
        rows=marine_rows,
        target_fields=[
            "time",
            "wave_height",
            "wave_period",
            "sea_surface_temperature",
        ],
    )

    print(
        f"Marine data berhasil masuk: "
        f"{len(marine_rows)} baris"
    )


    # =====================================================
    # 11. VALIDASI JUMLAH DATA
    # =====================================================

    print("\n=== VALIDASI DATABASE ===")

    weather_count = hook.get_first(
        "SELECT COUNT(*) FROM weather_data"
    )[0]

    air_count = hook.get_first(
        "SELECT COUNT(*) FROM air_quality_data"
    )[0]

    marine_count = hook.get_first(
        "SELECT COUNT(*) FROM marine_data"
    )[0]

    print(
        f"weather_data     : {weather_count} baris"
    )

    print(
        f"air_quality_data : {air_count} baris"
    )

    print(
        f"marine_data      : {marine_count} baris"
    )


    # =====================================================
    # 12. SELESAI
    # =====================================================

    print("\n========================================")
    print("ETL SELESAI")
    print("========================================")

    print(
        "Ketiga dataset disimpan secara terpisah."
    )

    print(
        "Assessing, cleaning, re-assessing, dan merging "
        "akan dilakukan di Google Colab."
    )


# =========================================================
# DAG
# =========================================================

with DAG(
    dag_id="weather_ml_etl",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["etl", "weather", "ml"],
) as dag:

    process_data = PythonOperator(
        task_id="extract_transform_load",
        python_callable=extract_transform_load,
    )