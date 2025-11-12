import os
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

def unzip_all():
    targets = {
        "roads": os.path.join(DATA_DIR, "processed/roads/ube_roads.zip"),
        "shelters": os.path.join(DATA_DIR, "processed/shelters/ube_shelters.zip"),
        "university": os.path.join(DATA_DIR, "raw/university/ube_university.zip"),
    }

    for name, zip_path in targets.items():
        if not os.path.exists(zip_path):
            print(f"⚠️ {name} ZIP not found: {zip_path}")
            continue

        extract_dir = os.path.dirname(zip_path)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"✅ Extracted {name}: {extract_dir}")
        except Exception as e:
            print(f"❌ Failed to extract {name}: {e}")

if __name__ == "__main__":
    unzip_all()