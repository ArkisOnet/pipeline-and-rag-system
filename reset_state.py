"""Reset all protocol statuses back to pending so the pipeline re-processes them."""
import sqlite3

conn = sqlite3.connect("data/state.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM protocol_urls WHERE status='done'")
count = cur.fetchone()[0]
print(f"Resetting {count} protocols to pending...")

cur.execute("UPDATE protocol_urls SET status='pending', error_msg=NULL")
conn.commit()

cur.execute("SELECT status, COUNT(*) FROM protocol_urls GROUP BY status")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
print("Done. Run: python -m pipeline.orchestrator --resume")
