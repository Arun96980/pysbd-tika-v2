import json

with open("./faiss_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

missing = [entry for entry in data if "sentence_hash" not in entry]

print(f"⚠️ Found {len(missing)} entries without sentence_hash")
for m in missing[:5]:  # just show first 5
    print(m)
