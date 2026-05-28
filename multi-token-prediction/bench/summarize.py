import json, glob, sys
patterns = sys.argv[1:] if len(sys.argv) > 1 else ["metrics/runs/*code*/result.json", "metrics/runs/*structured*/result.json", "metrics/runs/*baseline*/result.json", "metrics/runs/*mtp_n*/result.json"]
seen = set()
for pat in patterns:
    for p in sorted(glob.glob(pat)):
        if p in seen: continue
        seen.add(p)
        a = json.load(open(p))["aggregate"]
        if "mbu" not in a:
            continue
        s = a.get("speculative_decoding", {})
        thru = a["system_throughput_tokens_per_sec"]
        tpot = a["mbu"]["tpot_seconds"] * 1000
        mbu = a["mbu"]["mbu_fraction"] * 100
        accept = s.get("overall_acceptance_rate", 0) * 100
        prop = s.get("total_proposed_tokens", 0)
        compl = a["total_completion_tokens"]
        wall = a["wall_seconds"]
        print(p)
        print(f"  thru={thru:.2f} tok/s | tpot={tpot:.1f}ms | mbu={mbu:.2f}% | accept={accept:.2f}% | prop={prop} | compl_total={compl} | wall={wall:.1f}s")
