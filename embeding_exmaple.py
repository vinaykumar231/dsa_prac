"""
Simple ChromaDB Demo - No files needed!
Shows how to store and search multi-source embeddings
"""

import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# Initialize
print("🚀 Initializing...")
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./my_chroma_db")

# ============================================
# SAMPLE DATA
# ============================================

sample_data = [
    # PDF content
    {
        'content': 'Employee Safety Policy: All kitchen staff must wear non-slip shoes, aprons, and hair nets at all times.',
        'type': 'pdf',
        'source': 'employee_handbook.pdf',
        'page': '5'
    },
    {
        'content': 'Vacation Policy: Full-time employees receive 15 days of paid vacation annually after one year of service.',
        'type': 'pdf',
        'source': 'employee_handbook.pdf',
        'page': '12'
    },
    # Website content
    {
        'content': 'Veggie Burger: Plant-based burger with black beans and quinoa. Served with sweet potato fries. $12.99',
        'type': 'website',
        'source': 'www.restaurant.com/menu',
        'section': 'entrees'
    },
    {
        'content': 'Customer Review: Best vegetarian burger in town! Fresh ingredients and amazing flavor. Highly recommended.',
        'type': 'website',
        'source': 'www.restaurant.com/reviews',
        'section': 'reviews'
    },
    # Excel content
    {
        'content': 'Item: Black Beans. Current Stock: 50kg. Minimum: 20kg. Status: Sufficient. Updated: 2025-10-09',
        'type': 'excel',
        'source': 'inventory.xlsx',
        'row': '2'
    },
    {
        'content': 'Item: Quinoa. Current Stock: 8kg. Minimum: 15kg. Status: LOW - Reorder Needed. Updated: 2025-10-09',
        'type': 'excel',
        'source': 'inventory.xlsx',
        'row': '3'
    }
]

# ============================================
# CREATE COLLECTION & ADD DATA
# ============================================

print("📦 Creating ChromaDB collection...")

# Delete if exists (for clean demo)
try:
    client.delete_collection("demo_collection")
except:
    pass

# Create collection
collection = client.create_collection(
    name="demo_collection",
    metadata={"description": "Demo multi-source data"}
)

print("✨ Adding documents to ChromaDB...")

# Prepare data
documents = []
embeddings = []
ids = []
metadatas = []

for item in sample_data:
    # Add source type prefix
    full_text = f"[{item['type'].upper()}] {item['content']}"
    
    # Generate embedding
    embedding = model.encode(full_text).tolist()
    
    # Prepare for ChromaDB
    documents.append(item['content'])
    embeddings.append(embedding)
    ids.append(str(uuid.uuid4()))
    metadatas.append({
        'type': item['type'],
        'source': item['source']
    })

# Add to ChromaDB
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print(f"✓ Added {len(documents)} documents to ChromaDB")
print(f"✓ Data persisted to: ./my_chroma_db")

# ============================================
# SEARCH EXAMPLES
# ============================================

print("\n" + "="*70)
print("🔍 SEARCH EXAMPLES")
print("="*70)

def search(query, top_k=3, filter_type=None):
    """Search the collection"""
    query_embedding = model.encode(query).tolist()
    
    # Build filter
    where_filter = {"type": filter_type} if filter_type else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )
    
    return results

# Query 1: General search
print("\n📝 Query 1: 'safety rules'")
print("-" * 70)
results = search("safety rules", top_k=3)

for i in range(len(results['documents'][0])):
    print(f"\n{i+1}. [{results['metadatas'][0][i]['type'].upper()}]")
    print(f"   Distance: {results['distances'][0][i]:.4f}")
    print(f"   Source: {results['metadatas'][0][i]['source']}")
    print(f"   Content: {results['documents'][0][i]}")

# Query 2: Vegetarian food
print("\n\n📝 Query 2: 'vegetarian menu options'")
print("-" * 70)
results = search("vegetarian menu options", top_k=2)

for i in range(len(results['documents'][0])):
    print(f"\n{i+1}. [{results['metadatas'][0][i]['type'].upper()}]")
    print(f"   {results['documents'][0][i]}")

# Query 3: Inventory with filtering
print("\n\n📝 Query 3: 'ingredient levels' (Excel only)")
print("-" * 70)
results = search("ingredient levels", top_k=3, filter_type="excel")

for i in range(len(results['documents'][0])):
    print(f"\n{i+1}. {results['documents'][0][i]}")

# Query 4: Multi-answer query
print("\n\n📝 Query 4: 'can we make veggie burgers?'")
print("-" * 70)
results = search("can we make veggie burgers?", top_k=4)

print("\nTop results from different sources:")
for i in range(len(results['documents'][0])):
    print(f"\n{i+1}. [{results['metadatas'][0][i]['type'].upper()}]")
    print(f"   {results['documents'][0][i][:100]}...")

# ============================================
# COLLECTION STATS
# ============================================

print("\n\n" + "="*70)
print("📊 COLLECTION STATISTICS")
print("="*70)

count = collection.count()
print(f"\nTotal documents: {count}")

# Get all docs to count by type
all_docs = collection.get()
type_counts = {}
for metadata in all_docs['metadatas']:
    doc_type = metadata['type']
    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

print("\nDocuments by type:")
for doc_type, count in type_counts.items():
    print(f"  - {doc_type.upper()}: {count}")

print("\nSources:")
sources = set(m['source'] for m in all_docs['metadatas'])
for source in sources:
    print(f"  - {source}")

# ============================================
# PERSISTENCE DEMO
# ============================================

print("\n" + "="*70)
print("💾 PERSISTENCE DEMO")
print("="*70)

print("\nℹ️  Your data is saved to disk at: ./my_chroma_db")
print("\n   Next time you run this script:")
print("   1. Comment out the 'delete_collection' line")
print("   2. Use: collection = client.get_collection('demo_collection')")
print("   3. Your data will still be there!")

# ============================================
# ADD MORE DATA EXAMPLE
# ============================================

print("\n\n➕ Adding more data...")

new_data = {
    'content': 'Special Menu: Gluten-free pasta available. Ask your server for details.',
    'type': 'website',
    'source': 'www.restaurant.com/menu'
}

new_embedding = model.encode(f"[WEBSITE] {new_data['content']}").tolist()

collection.add(
    documents=[new_data['content']],
    embeddings=[new_embedding],
    ids=[str(uuid.uuid4())],
    metadatas=[{'type': new_data['type'], 'source': new_data['source']}]
)

print(f"✓ Added 1 new document. Total now: {collection.count()}")

# Search for it
print("\n🔍 Searching for new data: 'gluten free'")
results = search("gluten free", top_k=1)
print(f"   Found: {results['documents'][0][0]}")

# ============================================
# CLEANUP OPTION
# ============================================

print("\n\n" + "="*70)
print("🧹 CLEANUP")
print("="*70)

cleanup = input("\nDelete collection? (y/n): ").lower()
if cleanup == 'y':
    client.delete_collection("demo_collection")
    print("✓ Collection deleted")
else:
    print("✓ Collection preserved for next run")

print("\n" + "="*70)
print("✅ DEMO COMPLETE!")
print("="*70)