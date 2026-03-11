import json, os
f = open('data/reddit_enriched_data.jsonl','r',encoding='utf-8')
items = [json.loads(l) for l in f if l.strip()]
f.close()
ids = [it['id'] for it in items]
print(f"Total: {len(ids)}")
# Check Fakeddit_600_800
d = 'data/02_processed/images/Fakeddit_600_800'
found = sum(1 for i in ids if os.path.isfile(os.path.join(d, f'{i}.jpg')))
print(f"Fakeddit_600_800: {found}")
# Check Reddit dirs
for run in range(1,11):
    rd = f'data/02_processed/images/Reddit_reddit_run_{run:03d}'
    if os.path.isdir(rd):
        cnt = sum(1 for i in ids if os.path.isfile(os.path.join(rd, f'{i}.jpg')))
        if cnt > 0:
            print(f"  Reddit_run_{run:03d}: {cnt}")
total = 0
for i in ids:
    for d2 in ['data/02_processed/images/Fakeddit_600_800'] + [f'data/02_processed/images/Reddit_reddit_run_{r:03d}' for r in range(1,11)]:
        if os.path.isfile(os.path.join(d2, f'{i}.jpg')):
            total += 1
            break
print(f"\nTotal with images: {total} / {len(ids)}")
