from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
import os
# --- Config ---
service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")  # << เปลี่ยนตรงนี้
index_name = os.getenv("AZURE_SEARCH_INDEX_NAME") # << เปลี่ยนตรงนี้
admin_key = os.getenv("AZURE_SEARCH_KEY")

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") # << เปลี่ยนตรงนี้

credential = AzureKeyCredential(admin_key)

# --- Init Client ---
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

# --- Step 1: ดึงทุก document ID ---
print("📥 Fetching document IDs...")
doc_ids = []
results = search_client.search(search_text="*", select=["id"], top=1000, include_total_count=True)

for result in results:
    doc_ids.append(result["id"])

print(f"🔎 Found {len(doc_ids)} documents.")

# --- Step 2: ลบเอกสารทั้งหมด ---
if doc_ids:
    print("🗑️ Deleting documents...")
    delete_actions = [{"@search.action": "delete", "id": doc_id} for doc_id in doc_ids]

    # Batch delete (แบ่งเป็นชุดละ 1000 ถ้ามีเยอะ)
    batch_size = 5000
    for i in range(0, len(delete_actions), batch_size):
        batch = delete_actions[i:i+batch_size]
        result = search_client.upload_documents(documents=batch)
        print(f"✅ Deleted batch {i//batch_size + 1}, status: {result[0].status_code if result else 'unknown'}")

    print("✅ All documents deleted.")
else:
    print("ℹ️ No documents to delete.")
