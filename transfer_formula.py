import json

output_path = "run/generator/output_10.0_0.7.json"
solvents_path = "emb_dict/solvents.json"
salts_path = "emb_dict/salts.json"

with open(output_path) as f:
    entries = json.load(f)
with open(solvents_path) as f:
    solvents = json.load(f)
with open(salts_path) as f:
    salts = json.load(f)

num_solvents = len(solvents)

# 用列表收集所有结果
results = []

for i, entry in enumerate(entries, 1):
    bow = entry["bow"]
    solv = [(solvents[j]["name"], bow[j]) for j in range(num_solvents)
            if bow[j] > 0]
    salt = [(salts[j]["name"], bow[num_solvents + j])
            for j in range(len(salts)) if bow[num_solvents + j] > 0]

    solv_text = " + ".join([f"{name}:{ratio:.4f}"
                            for name, ratio in solv]) if solv else "None"
    salt_text = " + ".join([f"{name}:{ratio:.4f}"
                            for name, ratio in salt]) if salt else "None"

    results.append({
        "index": i,
        "solvent": solv_text,
        "salt": salt_text,
        "temperature": entry["temperature"],
        "conductivity": entry["conductivity"],
        "anion_ratio": entry["anion_ratio"]
    })

with open("electrolyte_formula.json", "w") as f:
    json.dump(results, f, indent=2)
