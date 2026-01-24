import os
from datetime import datetime, timezone

def combine():
    src_dir = os.path.join("logs", "session")
    if not os.path.isdir(src_dir):
        print("No logs/session directory found")
        return
    parts = []
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith('.log') and ('preview' in fn or 'session_preview' in fn):
            with open(os.path.join(src_dir, fn), 'r', encoding='utf-8') as f:
                parts.append(f"==== {fn} ====" + "\n")
                parts.append(f.read())

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_name = os.path.join(src_dir, f"preview_{ts}.log")
    with open(out_name, 'w', encoding='utf-8') as out:
        out.write('\n'.join(parts))
    print("Wrote combined preview log:", out_name)

if __name__ == '__main__':
    combine()
